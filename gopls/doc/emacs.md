# Emacs

Use [lsp-mode]. gopls is built in now as a client, so no special config is necessary. You first must install `gopls` and put it somewhere in your `PATH`. Here is an example (assuming you are using [use-package]) to get you started:

```lisp
(use-package lsp-mode
  :commands (lsp lsp-deferred))

(add-hook 'go-mode-hook #'lsp-deferred)

;; optional - provides fancier overlays
(use-package lsp-ui
  :commands lsp-ui-mode)

;; if you use company-mode for completion (otherwise, complete-at-point works out of the box):
(use-package company-lsp
  :commands company-lsp)
```

Common errors:
- When prompted by Emacs for your project folder, if you are using modules you must select the module's root folder (i.e. the directory with the "go.mod"). If you are using GOPATH, select your $GOPATH as your folder.
- Emacs must have your environment set properly (PATH, GOPATH, etc). You can run `M-x getenv <RET> PATH <RET>` to see if your PATH is set in Emacs. If not, you can try starting Emacs from your terminal, using [this package][exec-path-from-shell], or moving your shell config from .bashrc into .bashenv (or .zshenv).
- Make sure `lsp-mode`, `lsp-ui` and `company-lsp` are up-to-date, and make sure `lsp-go` is _not_ installed.

To troubleshoot, look in the `*lsp-log*` buffer for errors.

[lsp-mode]: https://github.com/emacs-lsp/lsp-mode
[use-package]: https://github.com/jwiegley/use-package
[exec-path-from-shell]: https://github.com/purcell/exec-path-from-shell