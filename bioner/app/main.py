import json
import litserve as ls
import logging
from pathlib import Path
from argparse import ArgumentParser, ArgumentTypeError
from app.interfaces import NERRequest
from app.engines import build_engine
from app.routes_training import router as training_router

logging.basicConfig(level=logging.INFO)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise ArgumentTypeError("Boolean value expected.")

def read_model_metadata(model_dir: str) -> dict:
        metadata_path = Path(model_dir) / "metadata.json"

        if not metadata_path.exists():
            metadata = {
                "name": "Extraction model",
                "version": "1.0"
            }
        else:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            if 'name' not in metadata or 'version' not in metadata:
                raise ValueError("metadata.json must contain at least 'name' and 'version'")
        
        return metadata

class NERAPI(ls.LitAPI):
    def __init__(self, 
                 engine: str, 
                 model: str, 
                 adapter_model: str | None = None,
                 prompt_path: str | None = None,
                 use_gpu: bool = False):
        super().__init__()
        self.engine = engine
        self.model = model
        self.adapter_model = adapter_model
        self.prompt_path = prompt_path
        self.use_gpu = use_gpu

    def setup(self, device):
        self.model = build_engine(
            engine=self.engine, 
            model=self.model, 
            adapter_model=self.adapter_model, 
            prompt_path=self.prompt_path,
            use_gpu=self.use_gpu)

    def decode_request(self, request: NERRequest) -> dict:
        return {
            "medical_text": request.medical_text,
            "labels": request.labels or [],
        }

    def predict(self, inputs: dict) -> dict:
        return self.model.extract_entities(medical_text=inputs["medical_text"], 
                                           labels=inputs["labels"])

    def encode_response(self, output):
        return output

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--engine", # previously model_type
                        type=str,
                        choices=["huggingface", "gliner", "gliner2"],
                        help="Type of model to use: 'huggingface' for Hugging Face LLM models or 'gliner' for GLiNER model."
    )
    parser.add_argument("--model", # previously model_path
                        type=str,
                        help="Path to the model to use. (Huggingface path)"
                        )
    parser.add_argument("--adapter_model", # previously adapter_path
                        type=str,
                        help="Path to the LLM adapter to use (if any)."
                        )
    parser.add_argument("--prompt_path",
                        type=str,
                        help="Path to the prompts file to use (if any)."
                        )
    parser.add_argument("--use_gpu",
                        type=str2bool,
                        default=False,
                        help="Flag to use GPU for inference."
                        )
    parser.add_argument("--host",
                        type=str,
                        default="0.0.0.0",
                        help="Host to run the server on."
                        )
    parser.add_argument("--port",
                        type=int,
                        default=8000,
                        help="Port to run the server on."
                        )
    args = parser.parse_args()
    model_metadata = read_model_metadata(args.model)
    api = NERAPI(
        engine=args.engine,
        model=args.model,
        adapter_model=args.adapter_model,
        prompt_path=args.prompt_path,
        use_gpu=args.use_gpu
    )
    server = ls.LitServer(
        api, 
        accelerator="auto", 
        timeout=300, 
        api_path="/ner", 
        info_path="/model/info",
        model_metadata=model_metadata
    )
    server.app.include_router(training_router)
    server.run(host=args.host, port=args.port)
