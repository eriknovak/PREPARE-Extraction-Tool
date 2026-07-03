# CLAUDE.md — frontend

React 19 + Vite SPA for the extract → cluster → map workflow. See root `CLAUDE.md` for the cross-service picture.

<!-- docs:start -->
## Documentation

Human + agent docs live in [`docs/`](./docs/README.md) — read these before deep work:

- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) — system map + validated Mermaid diagram
- [docs/setup-and-run.md](./docs/setup-and-run.md) — env vars, commands, test projects
- [docs/workflow.md](./docs/workflow.md) — the extract → cluster → map workflow across pages
- [docs/background/](./docs/background/) — deep dives: [api-layer](./docs/background/api-layer.md), [state-and-hooks](./docs/background/state-and-hooks.md), [routing-and-auth](./docs/background/routing-and-auth.md)
- [docs/components.md](./docs/components.md) · [docs/deploy.md](./docs/deploy.md)

Stack conventions: [.claude/project-rules/react-vite.md](./.claude/project-rules/react-vite.md).
@.claude/project-rules/react-vite.md
<!-- docs:end -->

## Stack

React 19 · Vite 7 · TypeScript 5.8 (strict) · React Router 7 · Vitest 3 · Storybook 9 · ESLint 9 (flat) ·
Prettier. UI: @dnd-kit (drag-drop clustering), Recharts, FontAwesome, date-fns, classnames. CSS Modules.
Package manager: **npm** (`package-lock.json`).

## Design & styling

**`DESIGN.md` is the authority for all UI styling — read it before writing or editing CSS.** Design
tokens (color, spacing, radius, type) live in `src/index.css`; reference `var(--…)`, never raw hex.
Classes follow BEM (`block__element--modifier`); one `styles.module.css` per component. Label-category
colors use the deterministic `label1`–`label9` ramp (see DESIGN.md §8).

## Commands

```bash
npm install
npm run dev                 # Vite dev server; proxies /api → BACKEND_HOST (default http://localhost:8000)
npm run build               # tsc -b && vite build → dist/
npm run preview
npm run lint                # eslint .
npm run format              # prettier --write .   (format:check to verify)
npm run test                # vitest run (unit + storybook story tests)
npm run test:watch
npm run test:coverage
npm run test -- src/utils/__tests__/dateUtils.test.ts   # single test file
npm run storybook           # :6006
npm run build-storybook
```

## Config

- `vite.config.ts` — dev `/api` proxy to `BACKEND_HOST` (10-min timeout for long extractions); port derived
  from `FRONTEND_HOST`; plugins: react, `vite-plugin-svgr` (SVG → component), `vite-tsconfig-paths`.
  Vitest has two projects: **unit** (node env, `*.test.{ts,tsx}`) and **storybook** (browser/Playwright).
- Path aliases (`tsconfig.app.json`): `@/`, `@components/`, `@pages/`, `@hooks/`, `@api/`, `@types/`, `@assets/`.
- Prettier: double quotes, 2-space, `printWidth: 120`, `trailingComma: es5`.

## Architecture

- **Entry**: `src/main.tsx` → `src/pages/App/index.tsx` (router). Routes are `React.lazy` + `Suspense`.
  All routes except `/login` wrapped in `ProtectedRoute` (checks `useAuth().isAuthenticated`).
- **API** (`src/api/`): plain **fetch**, no axios/react-query. `client.ts` is the base wrapper —
  attaches `Authorization: Bearer`, auto-refreshes on 401 (concurrent requests queued during refresh),
  base path `/api/v1`. One module per domain (`datasets`, `vocabularies`, `clusters`, `mappings`,
  `sourceTerms`, `records`, `extraction`, `auth`, `monitoring`). File uploads use `XMLHttpRequest` for
  progress (10-min timeout).
- **State**: React Context for auth (`AuthProvider` / `useAuth`); everything else is component `useState`
  + custom hooks. No Redux/Zustand, no SWR. Data refetch is manual via callbacks passed to hooks.
- **Hooks** (`src/hooks/`): data loaders (`useDatasets`, `useRecords`, `useSourceTerms`, `useVocabularies`…)
  and `useExtractionPolling` (polls extraction job status ~500–1000ms; persists job id in sessionStorage to
  resume across reload), `useToast`, `usePageTitle`.
- **Pages** (`src/pages/`): the workflow — `Datasets` → `DatasetUpload` / `DatasetOverview` →
  `DatasetTermExtraction` → `DatasetTermClustering` (drag-drop) → `DatasetConceptMapping`; plus
  `Vocabularies` / `VocabularyDetail` / `VocabularyUpload`, `Monitor` (training/extraction dashboard),
  `UserProfile`, `Login`.
- **Components** (`src/components/`): reusable UI, each its own dir with `index.tsx`, `*.module.css`, and
  often `*.stories.ts` alongside.

## Gotchas

- JWT access + refresh tokens in localStorage; failed refresh → logout + redirect to `/login`.
- Stories live next to components; `npm run test` runs them in a real browser via Playwright — needs
  browsers installed for the storybook project.
- Production API host: relative `/api/v1` by default; override with `VITE_BACKEND_HOST` if needed.
