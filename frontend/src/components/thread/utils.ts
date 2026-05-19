import type { Message } from "@langchain/langgraph-sdk";

export function getContentString(content: Message["content"]): string {
  if (typeof content === "string") return content;
  const texts = content
    .filter((c): c is { type: "text"; text: string } => c.type === "text")
    .map((c) => c.text);
  return texts.join(" ");
}

export function getReasoningString(message: Message): string | null {
  const parts: string[] = [];

  if (typeof message.content !== "string") {
    for (const block of message.content) {
      const b = block as Record<string, any>;
      if (b.type === "reasoning" && typeof b.reasoning === "string") {
        parts.push(b.reasoning);
      }
    }
  }

  const reasoningFromKwargs = (message as Record<string, any>)
    ?.additional_kwargs?.reasoning_content;
  if (
    typeof reasoningFromKwargs === "string" &&
    reasoningFromKwargs.length > 0
  ) {
    parts.push(reasoningFromKwargs);
  }

  return parts.length > 0 ? parts.join("\n") : null;
}
