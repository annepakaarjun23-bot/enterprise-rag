import { memo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";

function extractLang(className?: string): string | null {
  if (!className) return null;
  const match = className.match(/language-(\w+)/);
  return match ? match[1] : null;
}

/**
 * Markdown renderer for assistant answers.
 *
 * Uses rehype-highlight so ```python blocks get real syntax coloring.
 * We take over the <pre> element to attach a data-lang attribute for the
 * language pill, and to keep hljs's own background from double-applying.
 */
export const Markdown = memo(function Markdown({ children }: { children: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
        components={{
          pre({ children: preChildren, ...props }) {
            // find the language from the nested <code className="language-...">
            let lang: string | null = null;
            if (
              preChildren &&
              typeof preChildren === "object" &&
              "props" in (preChildren as object)
            ) {
              const child = (preChildren as { props?: { className?: string } }).props;
              lang = extractLang(child?.className);
            }
            return (
              <pre data-lang={lang ?? undefined} {...props}>
                {preChildren}
              </pre>
            );
          },
          a({ children: linkChildren, ...props }) {
            return (
              <a {...props} target="_blank" rel="noreferrer">
                {linkChildren}
              </a>
            );
          },
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
});