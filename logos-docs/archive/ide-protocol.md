# Section 21: IDE protocol (archived)

Historical, non-normative text from logos-docs/PLAN.md lines 1255-1269 at commit 57475ed. The normative spec is logos-docs/SPEC.md; the build plan is logos-docs/IMPLEMENTATION.md.

**SECTION 21: IDE & LANGUAGE SERVER (`gopls`) PROTOCOL EXTENSIONS:**

To ensure VSCode, Cursor, Zed, and Neovim work out-of-the-box with `.lgo`, `.📜`, `.⚡`, and `.ʕ◔ϖ◔ʔ` files, `loGo/s` extends `gopls` (`golang.org/x/tools/gopls`) with 4 features:

1. **Inlay Hints for Compiler Lowering:**
   * Displays inferred union tags, `T?` flow-narrowed non-nil types (`/* narrowed: *User */`), and `escape` allocation tier badges (`[stack]`, `[side-arena]`, `[heap]`, `[.rodata]`) inline in the editor.
2. **Hover Comptime Expansion Preview:**
   * Hovering over `const for`, `const switch T`, or `$"..."` renders the exact unrolled Go code and pre-computed byte buffer size (`Fixed: 17B + dynamic`) in the hover tooltip.
3. **Instant Code Actions (`QuickFix`):**
   * Binds `betterlint`'s `autofix_candidates` directly to LSP `textDocument/codeAction`, allowing 1-click or save-time conversion of `if err != nil` to `?` and nil-guards to `?.` / `??`.
4. **Virtual `.gen.go` Synchronization:**
   * When a workspace contains mixed `.lgo` and `.go` packages, `gopls` resolves type definitions directly across both file types in memory without requiring disk emission of `.gen.go` files.

---

