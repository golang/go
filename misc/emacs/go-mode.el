;;; go-mode.el --- Major mode for the Go programming language

;; Copyright 2013 The Go Authors. All rights reserved.
;; Use of this source code is governed by a BSD-style
;; license that can be found in the LICENSE file.

(require 'cl)
(require 'etags)
(require 'ffap)
(require 'find-file)
(require 'ring)
(require 'url)

;; XEmacs compatibility guidelines
;; - Minimum required version of XEmacs: 21.5.32
;;   - Feature that cannot be backported: POSIX character classes in
;;     regular expressions
;;   - Functions that could be backported but won't because 21.5.32
;;     covers them: plenty.
;;   - Features that are still partly broken:
;;     - godef will not work correctly if multibyte characters are
;;       being used
;;     - Fontification will not handle unicode correctly
;;
;; - Do not use \_< and \_> regexp delimiters directly; use
;;   go--regexp-enclose-in-symbol
;;
;; - The character `_` must not be a symbol constituent but a
;;   character constituent
;;
;; - Do not use process-lines
;;
;; - Use go--old-completion-list-style when using a plain list as the
;;   collection for completing-read
;;
;; - Use go--position-bytes instead of position-bytes
(defmacro go--xemacs-p ()
  `(featurep 'xemacs))

;; Delete the current line without putting it in the kill-ring.
(defun go--delete-whole-line (&optional arg)
  ;; Derived from `kill-whole-line'.
  ;; ARG is defined as for that function.
  (setq arg (or arg 1))
  (if (and (> arg 0)
           (eobp)
           (save-excursion (forward-visible-line 0) (eobp)))
      (signal 'end-of-buffer nil))
  (if (and (< arg 0)
           (bobp)
           (save-excursion (end-of-visible-line) (bobp)))
      (signal 'beginning-of-buffer nil))
  (cond ((zerop arg)
         (delete-region (progn (forward-visible-line 0) (point))
                        (progn (end-of-visible-line) (point))))
        ((< arg 0)
         (delete-region (progn (end-of-visible-line) (point))
                        (progn (forward-visible-line (1+ arg))
                               (unless (bobp)
                                 (backward-char))
                               (point))))
        (t
         (delete-region (progn (forward-visible-line 0) (point))
                        (progn (forward-visible-line arg) (point))))))

;; declare-function is an empty macro that only byte-compile cares
;; about. Wrap in always false if to satisfy Emacsen without that
;; macro.
(if nil
    (declare-function go--position-bytes "go-mode" (point)))

;; XEmacs unfortunately does not offer position-bytes. We can fall
;; back to just using (point), but it will be incorrect as soon as
;; multibyte characters are being used.
(if (fboundp 'position-bytes)
    (defalias 'go--position-bytes #'position-bytes)
  (defun go--position-bytes (point) point))

(defun go--old-completion-list-style (list)
  (mapcar (lambda (x) (cons x nil)) list))

;; GNU Emacs 24 has prog-mode, older GNU Emacs and XEmacs do not, so
;; copy its definition for those.
(if (not (fboundp 'prog-mode))
    (define-derived-mode prog-mode fundamental-mode "Prog"
      "Major mode for editing source code."
      (set (make-local-variable 'require-final-newline) mode-require-final-newline)
      (set (make-local-variable 'parse-sexp-ignore-comments) t)
      (setq bidi-paragraph-direction 'left-to-right)))

(defun go--regexp-enclose-in-symbol (s)
  ;; XEmacs does not support \_<, GNU Emacs does. In GNU Emacs we make
  ;; extensive use of \_< to support unicode in identifiers. Until we
  ;; come up with a better solution for XEmacs, this solution will
  ;; break fontification in XEmacs for identifiers such as "typeµ".
  ;; XEmacs will consider "type" a keyword, GNU Emacs won't.

  (if (go--xemacs-p)
      (concat "\\<" s "\\>")
    (concat "\\_<" s "\\_>")))

;; Move up one level of parentheses.
(defun go-goto-opening-parenthesis (&optional legacy-unused)
  ;; The old implementation of go-goto-opening-parenthesis had an
  ;; optional argument to speed up the function. It didn't change the
  ;; function's outcome.

  ;; Silently fail if there's no matching opening parenthesis.
  (condition-case nil
      (backward-up-list)
    (scan-error nil)))


(defconst go-dangling-operators-regexp "[^-]-\\|[^+]\\+\\|[/*&><.=|^]")
(defconst go-identifier-regexp "[[:word:][:multibyte:]]+")
(defconst go-label-regexp go-identifier-regexp)
(defconst go-type-regexp "[[:word:][:multibyte:]*]+")
(defconst go-func-regexp (concat (go--regexp-enclose-in-symbol "func") "\\s *\\(" go-identifier-regexp "\\)"))
(defconst go-func-meth-regexp (concat
                               (go--regexp-enclose-in-symbol "func") "\\s *\\(?:(\\s *"
                               "\\(" go-identifier-regexp "\\s +\\)?" go-type-regexp
                               "\\s *)\\s *\\)?\\("
                               go-identifier-regexp
                               "\\)("))
(defconst go-builtins
  '("append" "cap"   "close"   "complex" "copy"
    "delete" "imag"  "len"     "make"    "new"
    "panic"  "print" "println" "real"    "recover")
  "All built-in functions in the Go language. Used for font locking.")

(defconst go-mode-keywords
  '("break"    "default"     "func"   "interface" "select"
    "case"     "defer"       "go"     "map"       "struct"
    "chan"     "else"        "goto"   "package"   "switch"
    "const"    "fallthrough" "if"     "range"     "type"
    "continue" "for"         "import" "return"    "var")
  "All keywords in the Go language.  Used for font locking.")

(defconst go-constants '("nil" "true" "false" "iota"))
(defconst go-type-name-regexp (concat "\\(?:[*(]\\)*\\(?:" go-identifier-regexp "\\.\\)?\\(" go-identifier-regexp "\\)"))

(defvar go-dangling-cache)
(defvar go-godoc-history nil)
(defvar go--coverage-current-file-name)

(defgroup go nil
  "Major mode for editing Go code"
  :group 'languages)

(defgroup go-cover nil
  "Options specific to `cover`"
  :group 'go)

(defcustom go-fontify-function-calls t
  "Fontify function and method calls if this is non-nil."
  :type 'boolean
  :group 'go)

(defcustom go-mode-hook nil
  "Hook called by `go-mode'."
  :type 'hook
  :group 'go)

(defcustom go-command "go"
  "The 'go' command.  Some users have multiple Go development
trees and invoke the 'go' tool via a wrapper that sets GOROOT and
GOPATH based on the current directory.  Such users should
customize this variable to point to the wrapper script."
  :type 'string
  :group 'go)

(defcustom gofmt-command "gofmt"
  "The 'gofmt' command.  Some users may replace this with 'goimports'
from https://github.com/bradfitz/goimports."
  :type 'string
  :group 'go)

(defcustom go-other-file-alist
  '(("_test\\.go\\'" (".go"))
    ("\\.go\\'" ("_test.go")))
  "See the documentation of `ff-other-file-alist' for details."
  :type '(repeat (list regexp (choice (repeat string) function)))
  :group 'go)

(defface go-coverage-untracked
  '((t (:foreground "#505050")))
  "Coverage color of untracked code."
  :group 'go-cover)

(defface go-coverage-0
  '((t (:foreground "#c00000")))
  "Coverage color for uncovered code."
  :group 'go-cover)
(defface go-coverage-1
  '((t (:foreground "#808080")))
  "Coverage color for covered code with weight 1."
  :group 'go-cover)
(defface go-coverage-2
  '((t (:foreground "#748c83")))
  "Coverage color for covered code with weight 2."
  :group 'go-cover)
(defface go-coverage-3
  '((t (:foreground "#689886")))
  "Coverage color for covered code with weight 3."
  :group 'go-cover)
(defface go-coverage-4
  '((t (:foreground "#5ca489")))
  "Coverage color for covered code with weight 4."
  :group 'go-cover)
(defface go-coverage-5
  '((t (:foreground "#50b08c")))
  "Coverage color for covered code with weight 5."
  :group 'go-cover)
(defface go-coverage-6
  '((t (:foreground "#44bc8f")))
  "Coverage color for covered code with weight 6."
  :group 'go-cover)
(defface go-coverage-7
  '((t (:foreground "#38c892")))
  "Coverage color for covered code with weight 7."
  :group 'go-cover)
(defface go-coverage-8
  '((t (:foreground "#2cd495")))
  "Coverage color for covered code with weight 8.
For mode=set, all covered lines will have this weight."
  :group 'go-cover)
(defface go-coverage-9
  '((t (:foreground "#20e098")))
  "Coverage color for covered code with weight 9."
  :group 'go-cover)
(defface go-coverage-10
  '((t (:foreground "#14ec9b")))
  "Coverage color for covered code with weight 10."
  :group 'go-cover)
(defface go-coverage-covered
  '((t (:foreground "#2cd495")))
  "Coverage color of covered code."
  :group 'go-cover)

(defvar go-mode-syntax-table
  (let ((st (make-syntax-table)))
    (modify-syntax-entry ?+  "." st)
    (modify-syntax-entry ?-  "." st)
    (modify-syntax-entry ?%  "." st)
    (modify-syntax-entry ?&  "." st)
    (modify-syntax-entry ?|  "." st)
    (modify-syntax-entry ?^  "." st)
    (modify-syntax-entry ?!  "." st)
    (modify-syntax-entry ?=  "." st)
    (modify-syntax-entry ?<  "." st)
    (modify-syntax-entry ?>  "." st)
    (modify-syntax-entry ?/ (if (go--xemacs-p) ". 1456" ". 124b") st)
    (modify-syntax-entry ?*  ". 23" st)
    (modify-syntax-entry ?\n "> b" st)
    (modify-syntax-entry ?\" "\"" st)
    (modify-syntax-entry ?\' "\"" st)
    (modify-syntax-entry ?`  "\"" st)
    (modify-syntax-entry ?\\ "\\" st)
    ;; It would be nicer to have _ as a symbol constituent, but that
    ;; would trip up XEmacs, which does not support the \_< anchor
    (modify-syntax-entry ?_  "w" st)

    st)
  "Syntax table for Go mode.")

(defun go--build-font-lock-keywords ()
  ;; we cannot use 'symbols in regexp-opt because GNU Emacs <24
  ;; doesn't understand that
  (append
   `((,(go--regexp-enclose-in-symbol (regexp-opt go-mode-keywords t)) . font-lock-keyword-face)
     (,(concat "\\(" (go--regexp-enclose-in-symbol (regexp-opt go-builtins t)) "\\)[[:space:]]*(") 1 font-lock-builtin-face)
     (,(go--regexp-enclose-in-symbol (regexp-opt go-constants t)) . font-lock-constant-face)
     (,go-func-regexp 1 font-lock-function-name-face)) ;; function (not method) name

   (if go-fontify-function-calls
       `((,(concat "\\(" go-identifier-regexp "\\)[[:space:]]*(") 1 font-lock-function-name-face) ;; function call/method name
         (,(concat "[^[:word:][:multibyte:]](\\(" go-identifier-regexp "\\))[[:space:]]*(") 1 font-lock-function-name-face)) ;; bracketed function call
     `((,go-func-meth-regexp 2 font-lock-function-name-face))) ;; method name

   `(
     ("\\(`[^`]*`\\)" 1 font-lock-multiline) ;; raw string literal, needed for font-lock-syntactic-keywords
     (,(concat (go--regexp-enclose-in-symbol "type") "[[:space:]]+\\([^[:space:]]+\\)") 1 font-lock-type-face) ;; types
     (,(concat (go--regexp-enclose-in-symbol "type") "[[:space:]]+" go-identifier-regexp "[[:space:]]*" go-type-name-regexp) 1 font-lock-type-face) ;; types
     (,(concat "[^[:word:][:multibyte:]]\\[\\([[:digit:]]+\\|\\.\\.\\.\\)?\\]" go-type-name-regexp) 2 font-lock-type-face) ;; Arrays/slices
     (,(concat "\\(" go-identifier-regexp "\\)" "{") 1 font-lock-type-face)
     (,(concat (go--regexp-enclose-in-symbol "map") "\\[[^]]+\\]" go-type-name-regexp) 1 font-lock-type-face) ;; map value type
     (,(concat (go--regexp-enclose-in-symbol "map") "\\[" go-type-name-regexp) 1 font-lock-type-face) ;; map key type
     (,(concat (go--regexp-enclose-in-symbol "chan") "[[:space:]]*\\(?:<-\\)?" go-type-name-regexp) 1 font-lock-type-face) ;; channel type
     (,(concat (go--regexp-enclose-in-symbol "\\(?:new\\|make\\)") "\\(?:[[:space:]]\\|)\\)*(" go-type-name-regexp) 1 font-lock-type-face) ;; new/make type
     ;; TODO do we actually need this one or isn't it just a function call?
     (,(concat "\\.\\s *(" go-type-name-regexp) 1 font-lock-type-face) ;; Type conversion
     (,(concat (go--regexp-enclose-in-symbol "func") "[[:space:]]+(" go-identifier-regexp "[[:space:]]+" go-type-name-regexp ")") 1 font-lock-type-face) ;; Method receiver
     (,(concat (go--regexp-enclose-in-symbol "func") "[[:space:]]+(" go-type-name-regexp ")") 1 font-lock-type-face) ;; Method receiver without variable name
     ;; Like the original go-mode this also marks compound literal
     ;; fields. There, it was marked as to fix, but I grew quite
     ;; accustomed to it, so it'll stay for now.
     (,(concat "^[[:space:]]*\\(" go-label-regexp "\\)[[:space:]]*:\\(\\S.\\|$\\)") 1 font-lock-constant-face) ;; Labels and compound literal fields
     (,(concat (go--regexp-enclose-in-symbol "\\(goto\\|break\\|continue\\)") "[[:space:]]*\\(" go-label-regexp "\\)") 2 font-lock-constant-face)))) ;; labels in goto/break/continue

(defconst go--font-lock-syntactic-keywords
  ;; Override syntax property of raw string literal contents, so that
  ;; backslashes have no special meaning in ``. Used in Emacs 23 or older.
  '((go--match-raw-string-literal
     (1 (7 . ?`))
     (2 (15 . nil))  ;; 15 = "generic string"
     (3 (7 . ?`)))))

(defvar go-mode-map
  (let ((m (make-sparse-keymap)))
    (define-key m "}" #'go-mode-insert-and-indent)
    (define-key m ")" #'go-mode-insert-and-indent)
    (define-key m "," #'go-mode-insert-and-indent)
    (define-key m ":" #'go-mode-insert-and-indent)
    (define-key m "=" #'go-mode-insert-and-indent)
    (define-key m (kbd "C-c C-a") #'go-import-add)
    (define-key m (kbd "C-c C-j") #'godef-jump)
    (define-key m (kbd "C-x 4 C-c C-j") #'godef-jump-other-window)
    (define-key m (kbd "C-c C-d") #'godef-describe)
    m)
  "Keymap used by Go mode to implement electric keys.")

(defun go-mode-insert-and-indent (key)
  "Invoke the global binding of KEY, then reindent the line."

  (interactive (list (this-command-keys)))
  (call-interactively (lookup-key (current-global-map) key))
  (indent-according-to-mode))

(defmacro go-paren-level ()
  `(car (syntax-ppss)))

(defmacro go-in-string-or-comment-p ()
  `(nth 8 (syntax-ppss)))

(defmacro go-in-string-p ()
  `(nth 3 (syntax-ppss)))

(defmacro go-in-comment-p ()
  `(nth 4 (syntax-ppss)))

(defmacro go-goto-beginning-of-string-or-comment ()
  `(goto-char (nth 8 (syntax-ppss))))

(defun go--backward-irrelevant (&optional stop-at-string)
  "Skips backwards over any characters that are irrelevant for
indentation and related tasks.

It skips over whitespace, comments, cases and labels and, if
STOP-AT-STRING is not true, over strings."

  (let (pos (start-pos (point)))
    (skip-chars-backward "\n\s\t")
    (if (and (save-excursion (beginning-of-line) (go-in-string-p)) (looking-back "`") (not stop-at-string))
        (backward-char))
    (if (and (go-in-string-p) (not stop-at-string))
        (go-goto-beginning-of-string-or-comment))
    (if (looking-back "\\*/")
        (backward-char))
    (if (go-in-comment-p)
        (go-goto-beginning-of-string-or-comment))
    (setq pos (point))
    (beginning-of-line)
    (if (or (looking-at (concat "^" go-label-regexp ":")) (looking-at "^[[:space:]]*\\(case .+\\|default\\):"))
        (end-of-line 0)
      (goto-char pos))
    (if (/= start-pos (point))
        (go--backward-irrelevant stop-at-string))
    (/= start-pos (point))))

(defun go--buffer-narrowed-p ()
  "Return non-nil if the current buffer is narrowed."
  (/= (buffer-size)
      (- (point-max)
         (point-min))))

(defun go--match-raw-string-literal (end)
  "Search for a raw string literal. Set point to the end of the
occurence found on success. Returns nil on failure."
  (when (search-forward "`" end t)
    (goto-char (match-beginning 0))
    (if (go-in-string-or-comment-p)
        (progn (goto-char (match-end 0))
               (go--match-raw-string-literal end))
      (when (looking-at "\\(`\\)\\([^`]*\\)\\(`\\)")
        (goto-char (match-end 0))
        t))))

(defun go-previous-line-has-dangling-op-p ()
  "Returns non-nil if the current line is a continuation line."
  (let* ((cur-line (line-number-at-pos))
         (val (gethash cur-line go-dangling-cache 'nope)))
    (if (or (go--buffer-narrowed-p) (equal val 'nope))
        (save-excursion
          (beginning-of-line)
          (go--backward-irrelevant t)
          (setq val (looking-back go-dangling-operators-regexp))
          (if (not (go--buffer-narrowed-p))
              (puthash cur-line val go-dangling-cache))))
    val))

(defun go--at-function-definition ()
  "Return non-nil if point is on the opening curly brace of a
function definition.

We do this by first calling (beginning-of-defun), which will take
us to the start of *some* function. We then look for the opening
curly brace of that function and compare its position against the
curly brace we are checking. If they match, we return non-nil."
  (if (= (char-after) ?\{)
      (save-excursion
        (let ((old-point (point))
              start-nesting)
          (beginning-of-defun)
          (when (looking-at "func ")
            (setq start-nesting (go-paren-level))
            (skip-chars-forward "^{")
            (while (> (go-paren-level) start-nesting)
              (forward-char)
              (skip-chars-forward "^{") 0)
            (if (and (= (go-paren-level) start-nesting) (= old-point (point)))
                t))))))

(defun go--indentation-for-opening-parenthesis ()
  "Return the semantic indentation for the current opening parenthesis.

If point is on an opening curly brace and said curly brace
belongs to a function declaration, the indentation of the func
keyword will be returned. Otherwise the indentation of the
current line will be returned."
  (save-excursion
    (if (go--at-function-definition)
        (progn
          (beginning-of-defun)
          (current-indentation))
      (current-indentation))))

(defun go-indentation-at-point ()
  (save-excursion
    (let (start-nesting)
      (back-to-indentation)
      (setq start-nesting (go-paren-level))

      (cond
       ((go-in-string-p)
        (current-indentation))
       ((looking-at "[])}]")
        (go-goto-opening-parenthesis)
        (if (go-previous-line-has-dangling-op-p)
            (- (current-indentation) tab-width)
          (go--indentation-for-opening-parenthesis)))
       ((progn (go--backward-irrelevant t) (looking-back go-dangling-operators-regexp))
        ;; only one nesting for all dangling operators in one operation
        (if (go-previous-line-has-dangling-op-p)
            (current-indentation)
          (+ (current-indentation) tab-width)))
       ((zerop (go-paren-level))
        0)
       ((progn (go-goto-opening-parenthesis) (< (go-paren-level) start-nesting))
        (if (go-previous-line-has-dangling-op-p)
            (current-indentation)
          (+ (go--indentation-for-opening-parenthesis) tab-width)))
       (t
        (current-indentation))))))

(defun go-mode-indent-line ()
  (interactive)
  (let (indent
        shift-amt
        (pos (- (point-max) (point)))
        (point (point))
        (beg (line-beginning-position)))
    (back-to-indentation)
    (if (go-in-string-or-comment-p)
        (goto-char point)
      (setq indent (go-indentation-at-point))
      (if (looking-at (concat go-label-regexp ":\\([[:space:]]*/.+\\)?$\\|case .+:\\|default:"))
          (decf indent tab-width))
      (setq shift-amt (- indent (current-column)))
      (if (zerop shift-amt)
          nil
        (delete-region beg (point))
        (indent-to indent))
      ;; If initial point was within line's indentation,
      ;; position after the indentation.  Else stay at same point in text.
      (if (> (- (point-max) pos) (point))
          (goto-char (- (point-max) pos))))))

(defun go-beginning-of-defun (&optional count)
  (setq count (or count 1))
  (let ((first t)
        failure)
    (dotimes (i (abs count))
      (while (and (not failure)
                  (or first (go-in-string-or-comment-p)))
        (if (>= count 0)
            (progn
              (go--backward-irrelevant)
              (if (not (re-search-backward go-func-meth-regexp nil t))
                  (setq failure t)))
          (if (looking-at go-func-meth-regexp)
              (forward-char))
          (if (not (re-search-forward go-func-meth-regexp nil t))
              (setq failure t)))
        (setq first nil)))
    (if (< count 0)
        (beginning-of-line))
    (not failure)))

(defun go-end-of-defun ()
  (let (orig-level)
    ;; It can happen that we're not placed before a function by emacs
    (if (not (looking-at "func"))
        (go-beginning-of-defun -1))
    (skip-chars-forward "^{")
    (forward-char)
    (setq orig-level (go-paren-level))
    (while (>= (go-paren-level) orig-level)
      (skip-chars-forward "^}")
      (forward-char))))

;;;###autoload
(define-derived-mode go-mode prog-mode "Go"
  "Major mode for editing Go source text.

This mode provides (not just) basic editing capabilities for
working with Go code. It offers almost complete syntax
highlighting, indentation that is almost identical to gofmt and
proper parsing of the buffer content to allow features such as
navigation by function, manipulation of comments or detection of
strings.

In addition to these core features, it offers various features to
help with writing Go code. You can directly run buffer content
through gofmt, read godoc documentation from within Emacs, modify
and clean up the list of package imports or interact with the
Playground (uploading and downloading pastes).

The following extra functions are defined:

- `gofmt'
- `godoc'
- `go-import-add'
- `go-remove-unused-imports'
- `go-goto-imports'
- `go-play-buffer' and `go-play-region'
- `go-download-play'
- `godef-describe' and `godef-jump'
- `go-coverage'

If you want to automatically run `gofmt' before saving a file,
add the following hook to your emacs configuration:

\(add-hook 'before-save-hook #'gofmt-before-save)

If you want to use `godef-jump' instead of etags (or similar),
consider binding godef-jump to `M-.', which is the default key
for `find-tag':

\(add-hook 'go-mode-hook (lambda ()
                          (local-set-key (kbd \"M-.\") #'godef-jump)))

Please note that godef is an external dependency. You can install
it with

go get code.google.com/p/rog-go/exp/cmd/godef


If you're looking for even more integration with Go, namely
on-the-fly syntax checking, auto-completion and snippets, it is
recommended that you look at goflymake
\(https://github.com/dougm/goflymake), gocode
\(https://github.com/nsf/gocode), go-eldoc
\(github.com/syohex/emacs-go-eldoc) and yasnippet-go
\(https://github.com/dominikh/yasnippet-go)"

  ;; Font lock
  (set (make-local-variable 'font-lock-defaults)
       '(go--build-font-lock-keywords))

  ;; Indentation
  (set (make-local-variable 'indent-line-function) #'go-mode-indent-line)

  ;; Comments
  (set (make-local-variable 'comment-start) "// ")
  (set (make-local-variable 'comment-end)   "")
  (set (make-local-variable 'comment-use-syntax) t)
  (set (make-local-variable 'comment-start-skip) "\\(//+\\|/\\*+\\)\\s *")

  (set (make-local-variable 'beginning-of-defun-function) #'go-beginning-of-defun)
  (set (make-local-variable 'end-of-defun-function) #'go-end-of-defun)

  (set (make-local-variable 'parse-sexp-lookup-properties) t)
  (if (boundp 'syntax-propertize-function)
      (set (make-local-variable 'syntax-propertize-function) #'go-propertize-syntax)
    (set (make-local-variable 'font-lock-syntactic-keywords)
         go--font-lock-syntactic-keywords)
    (set (make-local-variable 'font-lock-multiline) t))

  (set (make-local-variable 'go-dangling-cache) (make-hash-table :test 'eql))
  (add-hook 'before-change-functions (lambda (x y) (setq go-dangling-cache (make-hash-table :test 'eql))) t t)

  ;; ff-find-other-file
  (setq ff-other-file-alist 'go-other-file-alist)

  (setq imenu-generic-expression
        '(("type" "^type *\\([^ \t\n\r\f]*\\)" 1)
          ("func" "^func *\\(.*\\) {" 1)))
  (imenu-add-to-menubar "Index")

  ;; Go style
  (setq indent-tabs-mode t)

  ;; Handle unit test failure output in compilation-mode
  ;;
  ;; Note the final t argument to add-to-list for append, ie put these at the
  ;; *ends* of compilation-error-regexp-alist[-alist]. We want go-test to be
  ;; handled first, otherwise other elements will match that don't work, and
  ;; those alists are traversed in *reverse* order:
  ;; http://lists.gnu.org/archive/html/bug-gnu-emacs/2001-12/msg00674.html
  (when (and (boundp 'compilation-error-regexp-alist)
             (boundp 'compilation-error-regexp-alist-alist))
    (add-to-list 'compilation-error-regexp-alist 'go-test t)
    (add-to-list 'compilation-error-regexp-alist-alist
                 '(go-test . ("^\t+\\([^()\t\n]+\\):\\([0-9]+\\):? .*$" 1 2)) t)))

;;;###autoload
(add-to-list 'auto-mode-alist (cons "\\.go\\'" 'go-mode))

(defun go--apply-rcs-patch (patch-buffer)
  "Apply an RCS-formatted diff from PATCH-BUFFER to the current
buffer."
  (let ((target-buffer (current-buffer))
        ;; Relative offset between buffer line numbers and line numbers
        ;; in patch.
        ;;
        ;; Line numbers in the patch are based on the source file, so
        ;; we have to keep an offset when making changes to the
        ;; buffer.
        ;;
        ;; Appending lines decrements the offset (possibly making it
        ;; negative), deleting lines increments it. This order
        ;; simplifies the forward-line invocations.
        (line-offset 0))
    (save-excursion
      (with-current-buffer patch-buffer
        (goto-char (point-min))
        (while (not (eobp))
          (unless (looking-at "^\\([ad]\\)\\([0-9]+\\) \\([0-9]+\\)")
            (error "invalid rcs patch or internal error in go--apply-rcs-patch"))
          (forward-line)
          (let ((action (match-string 1))
                (from (string-to-number (match-string 2)))
                (len  (string-to-number (match-string 3))))
            (cond
             ((equal action "a")
              (let ((start (point)))
                (forward-line len)
                (let ((text (buffer-substring start (point))))
                  (with-current-buffer target-buffer
                    (decf line-offset len)
                    (goto-char (point-min))
                    (forward-line (- from len line-offset))
                    (insert text)))))
             ((equal action "d")
              (with-current-buffer target-buffer
                (go--goto-line (- from line-offset))
                (incf line-offset len)
                (go--delete-whole-line len)))
             (t
              (error "invalid rcs patch or internal error in go--apply-rcs-patch")))))))))

(defun gofmt ()
  "Formats the current buffer according to the gofmt tool."

  (interactive)
  (let ((tmpfile (make-temp-file "gofmt" nil ".go"))
        (patchbuf (get-buffer-create "*Gofmt patch*"))
        (errbuf (get-buffer-create "*Gofmt Errors*"))
        (coding-system-for-read 'utf-8)
        (coding-system-for-write 'utf-8))

    (with-current-buffer errbuf
      (setq buffer-read-only nil)
      (erase-buffer))
    (with-current-buffer patchbuf
      (erase-buffer))

    (write-region nil nil tmpfile)

    ;; We're using errbuf for the mixed stdout and stderr output. This
    ;; is not an issue because gofmt -w does not produce any stdout
    ;; output in case of success.
    (if (zerop (call-process gofmt-command nil errbuf nil "-w" tmpfile))
        (if (zerop (call-process-region (point-min) (point-max) "diff" nil patchbuf nil "-n" "-" tmpfile))
            (progn
              (kill-buffer errbuf)
              (message "Buffer is already gofmted"))
          (go--apply-rcs-patch patchbuf)
          (kill-buffer errbuf)
          (message "Applied gofmt"))
      (message "Could not apply gofmt. Check errors for details")
      (gofmt--process-errors (buffer-file-name) tmpfile errbuf))

    (kill-buffer patchbuf)
    (delete-file tmpfile)))


(defun gofmt--process-errors (filename tmpfile errbuf)
  ;; Convert the gofmt stderr to something understood by the compilation mode.
  (with-current-buffer errbuf
    (goto-char (point-min))
    (insert "gofmt errors:\n")
    (while (search-forward-regexp (concat "^\\(" (regexp-quote tmpfile) "\\):") nil t)
      (replace-match (file-name-nondirectory filename) t t nil 1))
    (compilation-mode)
    (display-buffer errbuf)))

;;;###autoload
(defun gofmt-before-save ()
  "Add this to .emacs to run gofmt on the current buffer when saving:
 (add-hook 'before-save-hook 'gofmt-before-save).

Note that this will cause go-mode to get loaded the first time
you save any file, kind of defeating the point of autoloading."

  (interactive)
  (when (eq major-mode 'go-mode) (gofmt)))

(defun godoc--read-query ()
  "Read a godoc query from the minibuffer."
  ;; Compute the default query as the symbol under the cursor.
  ;; TODO: This does the wrong thing for e.g. multipart.NewReader (it only grabs
  ;; half) but I see no way to disambiguate that from e.g. foobar.SomeMethod.
  (let* ((bounds (bounds-of-thing-at-point 'symbol))
         (symbol (if bounds
                     (buffer-substring-no-properties (car bounds)
                                                     (cdr bounds)))))
    (completing-read (if symbol
                         (format "godoc (default %s): " symbol)
                       "godoc: ")
                     (go--old-completion-list-style (go-packages)) nil nil nil 'go-godoc-history symbol)))

(defun godoc--get-buffer (query)
  "Get an empty buffer for a godoc query."
  (let* ((buffer-name (concat "*godoc " query "*"))
         (buffer (get-buffer buffer-name)))
    ;; Kill the existing buffer if it already exists.
    (when buffer (kill-buffer buffer))
    (get-buffer-create buffer-name)))

(defun godoc--buffer-sentinel (proc event)
  "Sentinel function run when godoc command completes."
  (with-current-buffer (process-buffer proc)
    (cond ((string= event "finished\n")  ;; Successful exit.
           (goto-char (point-min))
           (view-mode 1)
           (display-buffer (current-buffer) t))
          ((/= (process-exit-status proc) 0)  ;; Error exit.
           (let ((output (buffer-string)))
             (kill-buffer (current-buffer))
             (message (concat "godoc: " output)))))))

;;;###autoload
(defun godoc (query)
  "Show Go documentation for a query, much like M-x man."
  (interactive (list (godoc--read-query)))
  (unless (string= query "")
    (set-process-sentinel
     (start-process-shell-command "godoc" (godoc--get-buffer query)
                                  (concat "godoc " query))
     'godoc--buffer-sentinel)
    nil))

(defun godoc-at-point (point)
  "Show Go documentation for the identifier at POINT.

`godoc-at-point' requires godef to work.

Due to a limitation in godoc, it is not possible to differentiate
between functions and methods, which may cause `godoc-at-point'
to display more documentation than desired."
  ;; TODO(dominikh): Support executing godoc-at-point on a package
  ;; name.
  (interactive "d")
  (condition-case nil
      (let* ((output (godef--call point))
             (file (car output))
             (name-parts (split-string (cadr output) " "))
             (first (car name-parts)))
        (if (not (godef--successful-p file))
            (message "%s" (godef--error file))
          (godoc (format "%s %s"
                         (file-name-directory file)
                         (if (or (string= first "type") (string= first "const"))
                             (cadr name-parts)
                           (car name-parts))))))
    (file-error (message "Could not run godef binary"))))

(defun go-goto-imports ()
  "Move point to the block of imports.

If using

  import (
    \"foo\"
    \"bar\"
  )

it will move point directly behind the last import.

If using

  import \"foo\"
  import \"bar\"

it will move point to the next line after the last import.

If no imports can be found, point will be moved after the package
declaration."
  (interactive)
  ;; FIXME if there's a block-commented import before the real
  ;; imports, we'll jump to that one.

  ;; Generally, this function isn't very forgiving. it'll bark on
  ;; extra whitespace. It works well for clean code.
  (let ((old-point (point)))
    (goto-char (point-min))
    (cond
     ((re-search-forward "^import ()" nil t)
      (backward-char 1)
      'block-empty)
     ((re-search-forward "^import ([^)]+)" nil t)
      (backward-char 2)
      'block)
     ((re-search-forward "\\(^import \\([^\"]+ \\)?\"[^\"]+\"\n?\\)+" nil t)
      'single)
     ((re-search-forward "^[[:space:]\n]*package .+?\n" nil t)
      (message "No imports found, moving point after package declaration")
      'none)
     (t
      (goto-char old-point)
      (message "No imports or package declaration found. Is this really a Go file?")
      'fail))))

(defun go-play-buffer ()
  "Like `go-play-region', but acts on the entire buffer."
  (interactive)
  (go-play-region (point-min) (point-max)))

(defun go-play-region (start end)
  "Send the region to the Playground and stores the resulting
link in the kill ring."
  (interactive "r")
  (let* ((url-request-method "POST")
         (url-request-extra-headers
          '(("Content-Type" . "application/x-www-form-urlencoded")))
         (url-request-data
          (encode-coding-string
           (buffer-substring-no-properties start end)
           'utf-8))
         (content-buf (url-retrieve
                       "http://play.golang.org/share"
                       (lambda (arg)
                         (cond
                          ((equal :error (car arg))
                           (signal 'go-play-error (cdr arg)))
                          (t
                           (re-search-forward "\n\n")
                           (kill-new (format "http://play.golang.org/p/%s" (buffer-substring (point) (point-max))))
                           (message "http://play.golang.org/p/%s" (buffer-substring (point) (point-max)))))))))))

;;;###autoload
(defun go-download-play (url)
  "Downloads a paste from the playground and inserts it in a Go
buffer. Tries to look for a URL at point."
  (interactive (list (read-from-minibuffer "Playground URL: " (ffap-url-p (ffap-string-at-point 'url)))))
  (with-current-buffer
      (let ((url-request-method "GET") url-request-data url-request-extra-headers)
        (url-retrieve-synchronously (concat url ".go")))
    (let ((buffer (generate-new-buffer (concat (car (last (split-string url "/"))) ".go"))))
      (goto-char (point-min))
      (re-search-forward "\n\n")
      (copy-to-buffer buffer (point) (point-max))
      (kill-buffer)
      (with-current-buffer buffer
        (go-mode)
        (switch-to-buffer buffer)))))

(defun go-propertize-syntax (start end)
  (save-excursion
    (goto-char start)
    (while (search-forward "\\" end t)
      (put-text-property (1- (point)) (point) 'syntax-table (if (= (char-after) ?`) '(1) '(9))))))

(defun go-import-add (arg import)
  "Add a new import to the list of imports.

When called with a prefix argument asks for an alternative name
to import the package as.

If no list exists yet, one will be created if possible.

If an identical import has been commented, it will be
uncommented, otherwise a new import will be added."

  ;; - If there's a matching `// import "foo"`, uncomment it
  ;; - If we're in an import() block and there's a matching `"foo"`, uncomment it
  ;; - Otherwise add a new import, with the appropriate syntax
  (interactive
   (list
    current-prefix-arg
    (replace-regexp-in-string "^[\"']\\|[\"']$" "" (completing-read "Package: " (go--old-completion-list-style (go-packages))))))
  (save-excursion
    (let (as line import-start)
      (if arg
          (setq as (read-from-minibuffer "Import as: ")))
      (if as
          (setq line (format "%s \"%s\"" as import))
        (setq line (format "\"%s\"" import)))

      (goto-char (point-min))
      (if (re-search-forward (concat "^[[:space:]]*//[[:space:]]*import " line "$") nil t)
          (uncomment-region (line-beginning-position) (line-end-position))
        (case (go-goto-imports)
          ('fail (message "Could not find a place to add import."))
          ('block-empty
           (insert "\n\t" line "\n"))
          ('block
              (save-excursion
                (re-search-backward "^import (")
                (setq import-start (point)))
            (if (re-search-backward (concat "^[[:space:]]*//[[:space:]]*" line "$")  import-start t)
                (uncomment-region (line-beginning-position) (line-end-position))
              (insert "\n\t" line)))
          ('single (insert "import " line "\n"))
          ('none (insert "\nimport (\n\t" line "\n)\n")))))))

(defun go-root-and-paths ()
  (let* ((output (split-string (shell-command-to-string (concat go-command " env GOROOT GOPATH"))
                               "\n"))
         (root (car output))
         (paths (split-string (cadr output) ":")))
    (append (list root) paths)))

(defun go--string-prefix-p (s1 s2 &optional ignore-case)
  "Return non-nil if S1 is a prefix of S2.
If IGNORE-CASE is non-nil, the comparison is case-insensitive."
  (eq t (compare-strings s1 nil nil
                         s2 0 (length s1) ignore-case)))

(defun go--directory-dirs (dir)
  "Recursively return all subdirectories in DIR."
  (if (file-directory-p dir)
      (let ((dir (directory-file-name dir))
            (dirs '())
            (files (directory-files dir nil nil t)))
        (dolist (file files)
          (unless (member file '("." ".."))
            (let ((file (concat dir "/" file)))
              (if (file-directory-p file)
                  (setq dirs (append (cons file
                                           (go--directory-dirs file))
                                     dirs))))))
        dirs)
    '()))


(defun go-packages ()
  (sort
   (delete-dups
    (mapcan
     (lambda (topdir)
       (let ((pkgdir (concat topdir "/pkg/")))
         (mapcan (lambda (dir)
                   (mapcar (lambda (file)
                             (let ((sub (substring file (length pkgdir) -2)))
                               (unless (or (go--string-prefix-p "obj/" sub) (go--string-prefix-p "tool/" sub))
                                 (mapconcat #'identity (cdr (split-string sub "/")) "/"))))
                           (if (file-directory-p dir)
                               (directory-files dir t "\\.a$"))))
                 (if (file-directory-p pkgdir)
                     (go--directory-dirs pkgdir)))))
     (go-root-and-paths)))
   #'string<))

(defun go-unused-imports-lines ()
  ;; FIXME Technically, -o /dev/null fails in quite some cases (on
  ;; Windows, when compiling from within GOPATH). Practically,
  ;; however, it has the same end result: There won't be a
  ;; compiled binary/archive, and we'll get our import errors when
  ;; there are any.
  (reverse (remove nil
                   (mapcar
                    (lambda (line)
                      (if (string-match "^\\(.+\\):\\([[:digit:]]+\\): imported and not used: \".+\".*$" line)
                          (if (string= (file-truename (match-string 1 line)) (file-truename buffer-file-name))
                              (string-to-number (match-string 2 line)))))
                    (split-string (shell-command-to-string
                                   (concat go-command
                                           (if (string-match "_test\.go$" buffer-file-truename)
                                               " test -c"
                                             " build -o /dev/null"))) "\n")))))

(defun go-remove-unused-imports (arg)
  "Removes all unused imports. If ARG is non-nil, unused imports
will be commented, otherwise they will be removed completely."
  (interactive "P")
  (save-excursion
    (let ((cur-buffer (current-buffer)) flymake-state lines)
      (when (boundp 'flymake-mode)
        (setq flymake-state flymake-mode)
        (flymake-mode-off))
      (save-some-buffers nil (lambda () (equal cur-buffer (current-buffer))))
      (if (buffer-modified-p)
          (message "Cannot operate on unsaved buffer")
        (setq lines (go-unused-imports-lines))
        (dolist (import lines)
          (go--goto-line import)
          (beginning-of-line)
          (if arg
              (comment-region (line-beginning-position) (line-end-position))
            (go--delete-whole-line)))
        (message "Removed %d imports" (length lines)))
      (if flymake-state (flymake-mode-on)))))

(defun godef--find-file-line-column (specifier other-window)
  "Given a file name in the format of `filename:line:column',
visit FILENAME and go to line LINE and column COLUMN."
  (if (not (string-match "\\(.+\\):\\([0-9]+\\):\\([0-9]+\\)" specifier))
      ;; We've only been given a directory name
      (funcall (if other-window #'find-file-other-window #'find-file) specifier)
    (let ((filename (match-string 1 specifier))
          (line (string-to-number (match-string 2 specifier)))
          (column (string-to-number (match-string 3 specifier))))
      (with-current-buffer (funcall (if other-window #'find-file-other-window #'find-file) filename)
        (go--goto-line line)
        (beginning-of-line)
        (forward-char (1- column))
        (if (buffer-modified-p)
            (message "Buffer is modified, file position might not have been correct"))))))

(defun godef--call (point)
  "Call godef, acquiring definition position and expression
description at POINT."
  (if (go--xemacs-p)
      (error "godef does not reliably work in XEmacs, expect bad results"))
  (if (not (buffer-file-name (go--coverage-origin-buffer)))
      (error "Cannot use godef on a buffer without a file name")
    (let ((outbuf (get-buffer-create "*godef*")))
      (with-current-buffer outbuf
        (erase-buffer))
      (call-process-region (point-min)
                           (point-max)
                           "godef"
                           nil
                           outbuf
                           nil
                           "-i"
                           "-t"
                           "-f"
                           (file-truename (buffer-file-name (go--coverage-origin-buffer)))
                           "-o"
                           (number-to-string (go--position-bytes point)))
      (with-current-buffer outbuf
        (split-string (buffer-substring-no-properties (point-min) (point-max)) "\n")))))

(defun godef--successful-p (output)
  (not (or (string= "-" output)
           (string= "godef: no identifier found" output)
           (go--string-prefix-p "godef: no declaration found for " output)
           (go--string-prefix-p "error finding import path for " output))))

(defun godef--error (output)
  (cond
   ((godef--successful-p output)
    nil)
   ((string= "-" output)
    "godef: expression is not defined anywhere")
   (t
    output)))

(defun godef-describe (point)
  "Describe the expression at POINT."
  (interactive "d")
  (condition-case nil
      (let ((description (cdr (butlast (godef--call point) 1))))
        (if (not description)
            (message "No description found for expression at point")
          (message "%s" (mapconcat #'identity description "\n"))))
    (file-error (message "Could not run godef binary"))))

(defun godef-jump (point &optional other-window)
  "Jump to the definition of the expression at POINT."
  (interactive "d")
  (condition-case nil
      (let ((file (car (godef--call point))))
        (if (not (godef--successful-p file))
            (message "%s" (godef--error file))
          (push-mark)
          (ring-insert find-tag-marker-ring (point-marker))
          (godef--find-file-line-column file other-window)))
    (file-error (message "Could not run godef binary"))))

(defun godef-jump-other-window (point)
  (interactive "d")
  (godef-jump point t))

(defun go--goto-line (line)
  (goto-char (point-min))
  (forward-line (1- line)))

(defun go--line-column-to-point (line column)
  (save-excursion
    (go--goto-line line)
    (forward-char (1- column))
    (point)))

(defstruct go--covered
  start-line start-column end-line end-column covered count)

(defun go--coverage-file ()
  "Return the coverage file to use, either by reading it from the
current coverage buffer or by prompting for it."
  (if (boundp 'go--coverage-current-file-name)
      go--coverage-current-file-name
    (read-file-name "Coverage file: " nil nil t)))

(defun go--coverage-origin-buffer ()
  "Return the buffer to base the coverage on."
  (or (buffer-base-buffer) (current-buffer)))

(defun go--coverage-face (count divisor)
  "Return the intensity face for COUNT when using DIVISOR
to scale it to a range [0,10].

DIVISOR scales the absolute cover count to values from 0 to 10.
For DIVISOR = 0 the count will always translate to 8."
  (let* ((norm (cond
                ((= count 0)
                 -0.1) ;; Uncovered code, set to -0.1 so n becomes 0.
                ((= divisor 0)
                 0.8) ;; covermode=set, set to 0.8 so n becomes 8.
                (t
                 (/ (log count) divisor))))
         (n (1+ (floor (* norm 9))))) ;; Convert normalized count [0,1] to intensity [0,10]
    (concat "go-coverage-" (number-to-string n))))

(defun go--coverage-make-overlay (range divisor)
  "Create a coverage overlay for a RANGE of covered/uncovered
code. Uses DIVISOR to scale absolute counts to a [0,10] scale."
  (let* ((count (go--covered-count range))
         (face (go--coverage-face count divisor))
         (ov (make-overlay (go--line-column-to-point (go--covered-start-line range)
                                                     (go--covered-start-column range))
                           (go--line-column-to-point (go--covered-end-line range)
                                                     (go--covered-end-column range)))))

    (overlay-put ov 'face face)
    (overlay-put ov 'help-echo (format "Count: %d" count))))

(defun go--coverage-clear-overlays ()
  "Remove existing overlays and put a single untracked overlay
over the entire buffer."
  (remove-overlays)
  (overlay-put (make-overlay (point-min) (point-max))
               'face
               'go-coverage-untracked))

(defun go--coverage-parse-file (coverage-file file-name)
  "Parse COVERAGE-FILE and extract coverage information and
divisor for FILE-NAME."
  (let (ranges
        (max-count 0))
    (with-temp-buffer
      (insert-file-contents coverage-file)
      (go--goto-line 2) ;; Skip over mode
      (while (not (eobp))
        (let* ((parts (split-string (buffer-substring (point-at-bol) (point-at-eol)) ":"))
               (file (car parts))
               (rest (split-string (nth 1 parts) "[., ]")))

          (destructuring-bind
              (start-line start-column end-line end-column num count)
              (mapcar #'string-to-number rest)

            (when (string= (file-name-nondirectory file) file-name)
              (if (> count max-count)
                  (setq max-count count))
              (push (make-go--covered :start-line start-line
                                      :start-column start-column
                                      :end-line end-line
                                      :end-column end-column
                                      :covered (/= count 0)
                                      :count count)
                    ranges)))

          (forward-line)))

      (list ranges (if (> max-count 0) (log max-count) 0)))))

(defun go-coverage (&optional coverage-file)
  "Open a clone of the current buffer and overlay it with
coverage information gathered via go test -coverprofile=COVERAGE-FILE.

If COVERAGE-FILE is nil, it will either be inferred from the
current buffer if it's already a coverage buffer, or be prompted
for."
  (interactive)
  (let* ((cur-buffer (current-buffer))
         (origin-buffer (go--coverage-origin-buffer))
         (gocov-buffer-name (concat (buffer-name origin-buffer) "<gocov>"))
         (coverage-file (or coverage-file (go--coverage-file)))
         (ranges-and-divisor (go--coverage-parse-file
                              coverage-file
                              (file-name-nondirectory (buffer-file-name origin-buffer))))
         (cov-mtime (nth 5 (file-attributes coverage-file)))
         (cur-mtime (nth 5 (file-attributes (buffer-file-name origin-buffer)))))

    (if (< (float-time cov-mtime) (float-time cur-mtime))
        (message "Coverage file is older than the source file."))

    (with-current-buffer (or (get-buffer gocov-buffer-name)
                             (make-indirect-buffer origin-buffer gocov-buffer-name t))
      (set (make-local-variable 'go--coverage-current-file-name) coverage-file)

      (save-excursion
        (go--coverage-clear-overlays)
        (dolist (range (car ranges-and-divisor))
          (go--coverage-make-overlay range (cadr ranges-and-divisor))))

      (if (not (eq cur-buffer (current-buffer)))
          (display-buffer (current-buffer) #'display-buffer-reuse-window)))))

(provide 'go-mode)
