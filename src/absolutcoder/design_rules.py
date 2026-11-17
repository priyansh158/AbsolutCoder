"""Built-in design rules that keep the UI AbsolutCoder writes from looking machine-made.

They are added to the prompt of the main model and of every worker role that
edits files, in every project, so they apply whoever writes the UI.
"""

from __future__ import annotations

DESIGN_RULES = """## Built-in design rules for web UI

These rules hold whenever you write or change web UI (HTML, CSS, JSX, TSX, Vue, Svelte, Astro). They keep the result from looking machine-generated.

In an existing project, follow its own conventions first (its framework, tokens, class names and component library); apply these rules to what you add, and never restyle or rewrite files the task did not ask you to touch.

1. Every colour and every font goes through a token declared at `:root` (or a declared theme block, or the project's own token system). No literal colour or `font-family` past the token block.
2. Never `transition: all`. Never animate `width`, `height`, `top` or `left`; animate `transform` and `opacity`.
3. Never one italic word inside a roman heading.
4. `overflow-x: clip` on both `html` and `body`. Never `hidden`, never `100vw`.
5. Image-bearing grid tracks use `minmax(0, 1fr)`; display headings carry `overflow-wrap: break-word; min-width: 0`.
6. Every interactive element has a `:focus-visible` style that appears instantly.
7. Every animation has a `prefers-reduced-motion: reduce` fallback, and content is visible with scripting off.
8. Never an invented metric, testimonial, logo wall or case-study count. A fact nobody gave you is absent.
9. Never an eyebrow above a heading, and never a label in tracked capitals or small caps. Labels, captions, table heads, tags and nav links use the body face, a step smaller, in sentence or lower case, untracked.
10. Never middle-dot-joined meta strings, arrows glued to link text, or decorative status dots.
11. Never a purple-to-blue gradient, a card inside a card, an emoji as an icon, or a hand-drawn browser or OS window.
12. On a page with a hero: its headline, one sentence and primary action are all visible at 1280 x 800 without scrolling.
13. On a page: about 80 words on any one screen; the hero's text is the headline, one sentence and its actions.
14. On a page: the container is capped and centred, so at 1920 wide the gutters grow and the measure and grids do not.

Games, canvas apps and app or tool screens: skip rules 12 to 14. Rule 4 does not apply to a game board, canvas or fixed app shell that must clip or fill the viewport. In rule 7, a game or app that cannot work without scripting needs only a short visible message when scripting is off. In rule 11, emoji may be game pieces or content, never interface icons.

Before you hand back UI work, check the files you wrote or changed against these rules and fix what breaks them."""
