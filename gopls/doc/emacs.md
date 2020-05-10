# Emacs

Use [lsp-mode]. gopls is built in as a client. You first must install `gopls` and put it somewhere in your `PATH`. Here is a basic config to get you started (assuming you are using [use-package]):

```lisp
(use-package lsp-mode
  :ensure t
  :commands (lsp lsp-deferred)
  :hook (go-mode . lsp-deferred))

;; Set up before-save hooks to format buffer and add/delete imports.
;; Make sure you don't have other gofmt/goimports hooks enabled.
(defun lsp-go-install-save-hooks ()
  (add-hook 'before-save-hook #'lsp-format-buffer t t)
  (add-hook 'before-save-hook #'lsp-organize-imports t t))
(add-hook 'go-mode-hook #'lsp-go-install-save-hooks)

;; Optional - provides fancier overlays.
(use-package lsp-ui
  :ensure t
  :commands lsp-ui-mode)

;; Company mode is a standard completion package that works well with lsp-mode.
(use-package company
  :ensure t
  :config
  ;; Optionally enable completion-as-you-type behavior.
  (setq company-idle-delay 0)
  (setq company-minimum-prefix-length 1))

;; Optional - provides snippet support.
(use-package yasnippet
  :ensure t
  :commands yas-minor-mode
  :hook (go-mode . yas-minor-mode))
```

lsp-mode integrates with xref. By default `lsp-find-definition` is bound to `M-.`. To go back, use `M-,`. Explore other `lsp-*` commands (not everything is supported by gopls).

## Gopls Configuration

Stable gopls settings have first-class support in [lsp-mode]. For example, `(setq lsp-gopls-use-placeholders nil)` will disable placeholders in completion snippets. See [lsp-go] for a list of available variables.

Experimental settings can be configured via `lsp-register-custom-settings`:

```lisp
(lsp-register-custom-settings
 '(("gopls.completeUnimported" t t)
   ("gopls.staticcheck" t t)))
```

See [settings] for information about gopls settings.

Note that after changing settings you must restart gopls using e.g. `M-x lsp-restart-workspace`.

## Troubleshooting

Common errors:
- When prompted by Emacs for your project folder, if you are using modules you must select the module's root folder (i.e. the directory with the "go.mod"). If you are using GOPATH, select your $GOPATH as your folder.
- Emacs must have your environment set properly (PATH, GOPATH, etc). You can run `M-x getenv <RET> PATH <RET>` to see if your PATH is set in Emacs. If not, you can try starting Emacs from your terminal, using [this package][exec-path-from-shell], or moving your shell config from .bashrc into .bashenv (or .zshenv).
- Make sure `lsp-mode` and `lsp-ui` are up-to-date, also make sure `lsp-go` and `company-lsp` are _not_ installed.
- Look for errors in the `*lsp-log*` buffer.
- Ask for help in the #emacs channel on the [Gophers slack].

[lsp-mode]: https://github.com/emacs-lsp/lsp-mode
[use-package]: https://github.com/jwiegley/use-package
[exec-path-from-shell]: https://github.com/purcell/exec-path-from-shell
[settings]: settings.md
[lsp-go]: https://github.com/emacs-lsp/lsp-mode/blob/master/lsp-go.el
[Gophers slack]: https://invite.slack.golangbridge.org/
