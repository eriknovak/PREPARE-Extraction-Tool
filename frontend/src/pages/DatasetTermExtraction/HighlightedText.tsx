import React, { useMemo } from "react";
import classNames from "classnames";

import { getLabelColorClass } from "@/utils/labelColors";

import type { SourceTerm } from "@/types";

import styles from "./styles.module.css";

interface HighlightedTextProps {
  text: string;
  terms: SourceTerm[];
  labels: string[];
  focusedTermId?: number | null;
}

const HighlightedText: React.FC<HighlightedTextProps> = ({ text, terms, labels, focusedTermId }) => {
  const segments = useMemo(() => {
    if (!terms.length) {
      return [{ type: "text" as const, content: text }];
    }

    // Filter terms with valid positions and sort by start position
    const validTerms = terms
      .filter((t) => t.start_position !== null && t.end_position !== null)
      .sort((a, b) => (a.start_position ?? 0) - (b.start_position ?? 0));

    if (!validTerms.length) {
      return [{ type: "text" as const, content: text }];
    }

    const result: Array<{ type: "text"; content: string } | { type: "term"; content: string; term: SourceTerm }> = [];
    let lastEnd = 0;

    for (const term of validTerms) {
      const start = term.start_position ?? 0;
      const end = term.end_position ?? 0;

      // Skip overlapping terms
      if (start < lastEnd) continue;

      // Add text before this term
      if (start > lastEnd) {
        result.push({ type: "text", content: text.slice(lastEnd, start) });
      }

      // Add the highlighted term
      result.push({
        type: "term",
        content: text.slice(start, end),
        term,
      });

      lastEnd = end;
    }

    // Add remaining text
    if (lastEnd < text.length) {
      result.push({ type: "text", content: text.slice(lastEnd) });
    }

    return result;
  }, [text, terms]);

  return (
    <div className={styles['record-text']}>
      {segments.map((segment, idx) =>
        segment.type === "text" ? (
          <span key={idx}>{segment.content}</span>
        ) : (
          <span
            key={idx}
            data-term-id={segment.term.id}
            className={classNames(styles['highlighted-term'], styles[getLabelColorClass(segment.term.label, labels)], {
              [styles['highlighted-term--focused']]: focusedTermId === segment.term.id,
            })}
            title={
              segment.term.links && segment.term.links.length > 0
                ? `${segment.term.label}: ${segment.term.value}\nLinked to: ${segment.term.links
                    .map((l) =>
                      l.from_term_id === segment.term.id ? l.to_term_value : l.from_term_value
                    )
                    .join(", ")}`
                : `${segment.term.label}: ${segment.term.value}`
            }
          >
            {segment.content}
            {segment.term.links && segment.term.links.length > 0 && (
              <span className={styles['highlighted-term__link-badge']} aria-hidden="true">
                🔗
              </span>
            )}
          </span>
        )
      )}
    </div>
  );
};

export default HighlightedText;
