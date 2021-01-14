# Emacs

## Installing `gopls`

To use `gopls` with Emacs, you must first
[install the `gopls` binary](../README.md#installation) and ensure that the directory
containing the resulting binary (either `$(go env GOBIN)` or `$(go env
GOPATH)/bin`) is in your `PATH`.

## Choosing an Emacs LSP client

To use `gopls` with Emacs, you will need to choose and install an Emacs LSP
client package. Two popular client packages are [LSP Mode] and [Eglot].

LSP Mode takes a batteries-included approach, with many integrations enabled
“out of the box” and several additional behaviors provided by `lsp-mode` itself.

Eglot takes a minimally-intrusive approach, focusing on smooth integration with
other established packages. It provides a few of its own `eglot-` commands but
no additional keybindings by default.

Once you have selected which client you want to use, install it per the packages
instructions: see [Eglot 1-2-3](https://github.com/joaotavora/eglot#1-2-3) or
[LSP Mode Installation](https://emacs-lsp.github.io/lsp-mode/page/installation/).

## Common configuration

Both Eglot and LSP Mode can integrate with popular packages in the Emacs
ecosystem:

* The built-in [`xref`] package provides cross-references.
* The built-in [Flymake] package provides an on-the-fly diagnostic overlay.
* [Company] mode displays code completion candidates (with a richer UI than
  the built-in [`completion-at-point`]).

Eglot provides documentation using the built-in [ElDoc] minor mode, while LSP
Mode by default provides documentation using its own [`lsp-ui`] mode.

Eglot by default locates the project root using the [`project`] package. In LSP
Mode, this behavior can be configured using the `lsp-auto-guess-root` setting.

## Configuring LSP Mode

### Loading LSP Mode in `.emacs`

```elisp
(require 'lsp-mode)
(add-hook 'go-mode-hook #'lsp-deferred)

;; Set up before-save hooks to format buffer and add/delete imports.
;; Make sure you don't have other gofmt/goimports hooks enabled.
(defun lsp-go-install-save-hooks ()
  (add-hook 'before-save-hook #'lsp-format-buffer t t)
  (add-hook 'before-save-hook #'lsp-organize-imports t t))
(add-hook 'go-mode-hook #'lsp-go-install-save-hooks)
```

### Configuring `gopls` via LSP Mode

See [settings] for information about available gopls settings.

Stable gopls settings have corresponding configuration variables in `lsp-mode`.
For example, `(setq lsp-gopls-use-placeholders nil)` will disable placeholders
in completion snippets. See [`lsp-go`] for a list of available variables.

Experimental settings can be configured via `lsp-register-custom-settings`:

```lisp
(lsp-register-custom-settings
 '(("gopls.completeUnimported" t t)
   ("gopls.staticcheck" t t)))
```

Note that after changing settings you must restart gopls using e.g. `M-x
lsp-restart-workspace`.

## Configuring Eglot

### Configuring `project` for Go modules in `.emacs`

Eglot uses the built-in `project` package to identify the LSP workspace for a
newly-opened buffer. The `project` package does not natively know about `GOPATH`
or Go modules. Fortunately, you can give it a custom hook to tell it to look for
the nearest parent `go.mod` file (that is, the root of the Go module) as the
project root.

```elisp
(require 'project)

(defun project-find-go-module (dir)
  (when-let ((root (locate-dominating-file dir "go.mod")))
    (cons 'go-module root)))

(cl-defmethod project-root ((project (head go-module)))
  (cdr project))

(add-hook 'project-find-functions #'project-find-go-module)
```

### Loading Eglot in `.emacs`

```elisp
;; Optional: load other packages before eglot to enable eglot integrations.
(require 'company)
(require 'yasnippet)

(require 'go-mode)
(require 'eglot)
(add-hook 'go-mode-hook 'eglot-ensure)

;; Optional: install eglot-format-buffer as a save hook.
;; The depth of -10 places this before eglot's willSave notification,
;; so that that notification reports the actual contents that will be saved.
(defun eglot-format-buffer-on-save ()
  (add-hook 'before-save-hook #'eglot-format-buffer -10 t))
(add-hook 'go-mode-hook #'eglot-format-buffer-on-save)
```

### Configuring `gopls` via Eglot

See [settings] for information about available gopls settings.

LSP server settings are controlled by the `eglot-workspace-configuration`
variable, which can be set either globally in `.emacs` (as below) or in a
`.dir-locals.el` file in the project root.

```elisp
(setq-default eglot-workspace-configuration
    '((:gopls .
        ((staticcheck . t)
         (matcher . "CaseSensitive")))))
```

### Organizing imports with Eglot

`gopls` provides the import-organizing functionality of `goimports` as an LSP
code action, which you can invoke as needed by running `M-x eglot-code-actions`
(or a key of your choice bound to the `eglot-code-actions` function) and
selecting `Organize Imports` at the prompt.

Eglot does not currently support a standalone function to execute a specific
code action (see
[joaotavora/eglot#411](https://github.com/joaotavora/eglot/issues/411)), nor an
option to organize imports as a `before-save-hook` (see
[joaotavora/eglot#574](https://github.com/joaotavora/eglot/issues/574)). In the
meantime, see those issues for discussion and possible workarounds.

## Troubleshooting

Common errors:

* When prompted by Emacs for your project folder, if you are using modules you
  must select the module's root folder (i.e. the directory with the "go.mod").
  If you are using GOPATH, select your $GOPATH as your folder.
* Emacs must have your environment set properly (PATH, GOPATH, etc). You can
  run `M-x getenv <RET> PATH <RET>` to see if your PATH is set in Emacs. If
  not, you can try starting Emacs from your terminal, using [this
  package][exec-path-from-shell], or moving your shell config from `.bashrc`
  into `.profile` and logging out and back in.
* Make sure only one LSP client mode is installed. (For example, if using
  `lsp-mode`, ensure that you are not _also_ enabling `eglot`.)
* Look for errors in the `*lsp-log*` buffer or run `M-x eglot-events-buffer`.
* Ask for help in the `#emacs` channel on the [Gophers slack].

[LSP Mode]: https://emacs-lsp.github.io/lsp-mode/
[Eglot]: https://github.com/joaotavora/eglot/blob/master/README.md
[`xref`]: https://www.gnu.org/software/emacs/manual/html_node/emacs/Xref.html
[Flymake]: https://www.gnu.org/software/emacs/manual/html_node/flymake/Using-Flymake.html#Using-Flymake
[Company]: https://company-mode.github.io/
[`completion-at-point`]: https://www.gnu.org/software/emacs/manual/html_node/elisp/Completion-in-Buffers.html
[ElDoc]: https://elpa.gnu.org/packages/eldoc.html
[`lsp-ui`]: https://emacs-lsp.github.io/lsp-ui/
[`lsp-go`]: https://github.com/emacs-lsp/lsp-mode/blob/master/clients/lsp-go.el
[`use-package`]: https://github.com/jwiegley/use-package
[`exec-path-from-shell`]: https://github.com/purcell/exec-path-from-shell
[settings]: settings.md
[Gophers slack]: https://invite.slack.golangbridge.org/
