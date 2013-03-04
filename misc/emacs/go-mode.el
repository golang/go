;;; go-mode.el --- Major mode for the Go programming language

;; Copyright 2013 The Go Authors. All rights reserved.
;; Use of this source code is governed by a BSD-style
;; license that can be found in the LICENSE file.

(require 'cl)
(require 'diff-mode)
(require 'ffap)
(require 'find-lisp)
(require 'url)

(defconst go-dangling-operators-regexp "[^-]-\\|[^+]\\+\\|[/*&><.=|^]")
(defconst gofmt-stdin-tag "<standard input>")
(defconst go-identifier-regexp "[[:word:][:multibyte:]_]+")
(defconst go-label-regexp go-identifier-regexp)
(defconst go-type-regexp "[[:word:][:multibyte:]_*]+")
(defconst go-func-regexp (concat "\\<func\\>\\s *\\(" go-identifier-regexp "\\)"))
(defconst go-func-meth-regexp (concat "\\<func\\>\\s *\\(?:(\\s *" go-identifier-regexp "\\s +" go-type-regexp "\\s *)\\s *\\)?\\(" go-identifier-regexp "\\)("))
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

(defgroup go nil
  "Major mode for editing Go code"
  :group 'languages)

(defcustom go-fontify-function-calls t
  "Fontify function and method calls if this is non-nil."
  :type 'boolean
  :group 'go)

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
    (modify-syntax-entry ?/  ". 124b" st)
    (modify-syntax-entry ?*  ". 23" st)
    (modify-syntax-entry ?\n "> b" st)
    (modify-syntax-entry ?\" "\"" st)
    (modify-syntax-entry ?\' "\"" st)
    (modify-syntax-entry ?`  "\"" st)
    (modify-syntax-entry ?\\ "\\" st)
    (modify-syntax-entry ?_  "_" st)

    st)
  "Syntax table for Go mode.")

(defun go--build-font-lock-keywords ()
  (append
   `((,(regexp-opt go-mode-keywords 'symbols) . font-lock-keyword-face)
     (,(regexp-opt go-builtins 'symbols) . font-lock-builtin-face)
     (,(regexp-opt go-constants 'symbols) . font-lock-constant-face)
     (,go-func-regexp 1 font-lock-function-name-face)) ;; function (not method) name

   (if go-fontify-function-calls
       `((,(concat "\\(" go-identifier-regexp "\\)[[:space:]]*(") 1 font-lock-function-name-face) ;; function call/method name
         (,(concat "(\\(" go-identifier-regexp "\\))[[:space:]]*(") 1 font-lock-function-name-face)) ;; bracketed function call
     `((,go-func-meth-regexp 1 font-lock-function-name-face))) ;; method name

   `(
     ("\\<type\\>[[:space:]]*\\([^[:space:]]+\\)" 1 font-lock-type-face) ;; types
     (,(concat "\\<type\\>[[:space:]]*" go-identifier-regexp "[[:space:]]*" go-type-name-regexp) 1 font-lock-type-face) ;; types
     (,(concat "\\(?:[[:space:]]+\\|\\]\\)\\[\\([[:digit:]]+\\|\\.\\.\\.\\)?\\]" go-type-name-regexp) 2 font-lock-type-face) ;; Arrays/slices
     (,(concat "map\\[[^]]+\\]" go-type-name-regexp) 1 font-lock-type-face) ;; map value type
     (,(concat "\\(" go-identifier-regexp "\\)" "{") 1 font-lock-type-face)
     (,(concat "\\<map\\[" go-type-name-regexp) 1 font-lock-type-face) ;; map key type
     (,(concat "\\<chan\\>[[:space:]]*\\(?:<-\\)?" go-type-name-regexp) 1 font-lock-type-face) ;; channel type
     (,(concat "\\<\\(?:new\\|make\\)\\>\\(?:[[:space:]]\\|)\\)*(" go-type-name-regexp) 1 font-lock-type-face) ;; new/make type
     ;; TODO do we actually need this one or isn't it just a function call?
     (,(concat "\\.\\s *(" go-type-name-regexp) 1 font-lock-type-face) ;; Type conversion
     (,(concat "\\<func\\>[[:space:]]+(" go-identifier-regexp "[[:space:]]+" go-type-name-regexp ")") 1 font-lock-type-face) ;; Method receiver
     ;; Like the original go-mode this also marks compound literal
     ;; fields. There, it was marked as to fix, but I grew quite
     ;; accustomed to it, so it'll stay for now.
     (,(concat "^[[:space:]]*\\(" go-label-regexp "\\)[[:space:]]*:\\(\\S.\\|$\\)") 1 font-lock-constant-face) ;; Labels and compound literal fields
     (,(concat "\\<\\(goto\\|break\\|continue\\)\\>[[:space:]]*\\(" go-label-regexp "\\)") 2 font-lock-constant-face)))) ;; labels in goto/break/continue

(defvar go-mode-map
  (let ((m (make-sparse-keymap)))
    (define-key m "}" 'go-mode-insert-and-indent)
    (define-key m ")" 'go-mode-insert-and-indent)
    (define-key m "," 'go-mode-insert-and-indent)
    (define-key m ":" 'go-mode-insert-and-indent)
    (define-key m "=" 'go-mode-insert-and-indent)
    (define-key m (kbd "C-c C-a") 'go-import-add)
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
    (skip-chars-backward "\n[:blank:]")
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

(defun go-goto-opening-parenthesis (&optional char)
  (let ((start-nesting (go-paren-level)))
    (while (and (not (bobp))
                (>= (go-paren-level) start-nesting))
      (if (zerop (skip-chars-backward
                  (if char
                      (case char (?\] "^[") (?\} "^{") (?\) "^("))
                    "^[{(")))
          (if (go-in-string-or-comment-p)
              (go-goto-beginning-of-string-or-comment)
            (backward-char))))))

(defun go-indentation-at-point ()
  (save-excursion
    (let (start-nesting (outindent 0))
      (back-to-indentation)
      (setq start-nesting (go-paren-level))

      (cond
       ((go-in-string-p)
        (current-indentation))
       ((looking-at "[])}]")
        (go-goto-opening-parenthesis (char-after))
        (if (go-previous-line-has-dangling-op-p)
            (- (current-indentation) tab-width)
          (current-indentation)))
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
          (+ (current-indentation) tab-width)))
       (t
        (current-indentation))))))

(defun go-mode-indent-line ()
  (interactive)
  (let (indent
        shift-amt
        end
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
  (unless count (setq count 1))
  (let ((first t) failure)
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
(define-derived-mode go-mode fundamental-mode "Go"
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

If you want to automatically run `gofmt' before saving a file,
add the following hook to your emacs configuration:

\(add-hook 'before-save-hook 'gofmt-before-save)

If you're looking for even more integration with Go, namely
on-the-fly syntax checking, auto-completion and snippets, it is
recommended that you look at goflymake
\(https://github.com/dougm/goflymake), gocode
\(https://github.com/nsf/gocode) and yasnippet-go
\(https://github.com/dominikh/yasnippet-go)"

  ;; Font lock
  (set (make-local-variable 'font-lock-defaults)
       '(go--build-font-lock-keywords))

  ;; Indentation
  (set (make-local-variable 'indent-line-function) 'go-mode-indent-line)

  ;; Comments
  (set (make-local-variable 'comment-start) "// ")
  (set (make-local-variable 'comment-end)   "")
  (set (make-local-variable 'comment-use-syntax) t)
  (set (make-local-variable 'comment-start-skip) "\\(//+\\|/\\*+\\)\\s *")

  (set (make-local-variable 'beginning-of-defun-function) 'go-beginning-of-defun)
  (set (make-local-variable 'end-of-defun-function) 'go-end-of-defun)

  (set (make-local-variable 'parse-sexp-lookup-properties) t)
  (if (boundp 'syntax-propertize-function)
      (set (make-local-variable 'syntax-propertize-function) 'go-propertize-syntax))

  (set (make-local-variable 'go-dangling-cache) (make-hash-table :test 'eql))
  (add-hook 'before-change-functions (lambda (x y) (setq go-dangling-cache (make-hash-table :test 'eql))) t t)


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

(defun gofmt ()
  "Pipe the current buffer through the external tool `gofmt`.
Replace the current buffer on success; display errors on failure."

  (interactive)
  (let ((currconf (current-window-configuration)))
    (let ((srcbuf (current-buffer))
          (filename buffer-file-name)
          (patchbuf (get-buffer-create "*Gofmt patch*")))
      (with-current-buffer patchbuf
        (let ((errbuf (get-buffer-create "*Gofmt Errors*"))
              ;; use utf-8 with subprocesses
              (coding-system-for-read 'utf-8)
              (coding-system-for-write 'utf-8))
          (with-current-buffer errbuf
            (setq buffer-read-only nil)
            (erase-buffer))
          (with-current-buffer srcbuf
            (save-restriction
              (let (deactivate-mark)
                (widen)
                ;; If this is a new file, diff-mode can't apply a
                ;; patch to a non-exisiting file, so replace the buffer
                ;; completely with the output of 'gofmt'.
                ;; If the file exists, patch it to keep the 'undo' list happy.
                (let* ((newfile (not (file-exists-p filename)))
                       (flag (if newfile "" " -d")))

                  ;; diff-mode doesn't work too well with missing
                  ;; end-of-file newline, so add one
                  (if (/= (char-after (1- (point-max))) ?\n)
                      (save-excursion
                        (goto-char (point-max))
                        (insert ?\n)))

                  (if (zerop (shell-command-on-region (point-min) (point-max)
                                                      (concat "gofmt" flag)
                                                      patchbuf nil errbuf))
                      ;; gofmt succeeded: replace buffer or apply patch hunks.
                      (let ((old-point (point))
                            (old-mark (mark t)))
                        (kill-buffer errbuf)
                        (if newfile
                            ;; New file, replace it (diff-mode won't work)
                            (gofmt--replace-buffer srcbuf patchbuf)
                          ;; Existing file, patch it
                          (gofmt--apply-patch filename srcbuf patchbuf))
                        (goto-char (min old-point (point-max)))
                        ;; Restore the mark and point
                        (if old-mark (push-mark (min old-mark (point-max)) t))
                        (set-window-configuration currconf))

                    ;; gofmt failed: display the errors
                    (message "Could not apply gofmt. Check errors for details")
                    (gofmt--process-errors filename errbuf))))))

          ;; Collapse any window opened on outbuf if shell-command-on-region
          ;; displayed it.
          (delete-windows-on patchbuf)))
      (kill-buffer patchbuf))))

(defun gofmt--replace-buffer (srcbuf patchbuf)
  (with-current-buffer srcbuf
    (erase-buffer)
    (insert-buffer-substring patchbuf))
  (message "Applied gofmt"))

(defun gofmt--apply-patch (filename srcbuf patchbuf)
  ;; apply all the patch hunks
  (let (changed)
    (with-current-buffer patchbuf
      (goto-char (point-min))
      ;; The .* is for TMPDIR, but to avoid dealing with TMPDIR
      ;; having a trailing / or not, it's easier to just search for .*
      ;; especially as we're only replacing the first instance.
      (if (re-search-forward "^--- \\(.*/gofmt[0-9]*\\)" nil t)
          (replace-match filename nil nil nil 1))
      (condition-case nil
          (while t
            (diff-hunk-next)
            (diff-apply-hunk)
            (setq changed t))
        ;; When there's no more hunks, diff-hunk-next signals an error, ignore it
        (error nil)))
    (if changed (message "Applied gofmt") (message "Buffer was already gofmted"))))

(defun gofmt--process-errors (filename errbuf)
  ;; Convert the gofmt stderr to something understood by the compilation mode.
  (with-current-buffer errbuf
    (goto-char (point-min))
    (insert "gofmt errors:\n")
    (if (search-forward gofmt-stdin-tag nil t)
        (replace-match (file-name-nondirectory filename) nil t))
    (display-buffer errbuf)
    (compilation-mode)))

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
                     (go-packages) nil nil nil 'go-godoc-history symbol)))

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
  "Show go documentation for a query, much like M-x man."
  (interactive (list (godoc--read-query)))
  (unless (string= query "")
    (set-process-sentinel
     (start-process-shell-command "godoc" (godoc--get-buffer query)
                                  (concat "godoc " query))
     'godoc--buffer-sentinel)
    nil))

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
         (url-request-data (buffer-substring-no-properties start end))
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

;; ;; Commented until we actually make use of this function
;; (defun go--common-prefix (sequences)
;;   ;; mismatch and reduce are cl
;;   (assert sequences)
;;   (flet ((common-prefix (s1 s2)
;;                         (let ((diff-pos (mismatch s1 s2)))
;;                           (if diff-pos (subseq s1 0 diff-pos) s1))))
;;     (reduce #'common-prefix sequences)))

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
    (replace-regexp-in-string "^[\"']\\|[\"']$" "" (completing-read "Package: " (go-packages)))))
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
  (let* ((output (process-lines "go" "env" "GOROOT" "GOPATH"))
         (root (car output))
         (paths (split-string (car (cdr output)) ":")))
    (append (list root) paths)))

(defun go-packages ()
  (sort
   (delete-dups
    (mapcan
     (lambda (topdir)
       (let ((pkgdir (concat topdir "/pkg/")))
         (mapcan (lambda (dir)
                   (mapcar (lambda (file)
                             (let ((sub (substring file (length pkgdir) -2)))
                               (unless (or (string-prefix-p "obj/" sub) (string-prefix-p "tool/" sub))
                                 (mapconcat 'identity (cdr (split-string sub "/")) "/"))))
                           (if (file-directory-p dir)
                               (directory-files dir t "\\.a$"))))
                 (if (file-directory-p pkgdir)
                     (find-lisp-find-files-internal pkgdir 'find-lisp-file-predicate-is-directory 'find-lisp-default-directory-predicate)))))
     (go-root-and-paths)))
   'string<))

(defun go-unused-imports-lines ()
  ;; FIXME Technically, -o /dev/null fails in quite some cases (on
  ;; Windows, when compiling from within GOPATH). Practically,
  ;; however, it has the same end result: There won't be a
  ;; compiled binary/archive, and we'll get our import errors when
  ;; there are any.
  (reverse (remove nil
                   (mapcar
                    (lambda (line)
                      (if (string-match "^\\(.+\\):\\([[:digit:]]+\\): imported and not used: \".+\"$" line)
                          (if (string= (file-truename (match-string 1 line)) (file-truename buffer-file-name))
                              (string-to-number (match-string 2 line)))))
                    (split-string (shell-command-to-string
                                   (if (string-match "_test\.go$" buffer-file-truename)
                                       "go test -c"
                                     "go build -o /dev/null")) "\n")))))

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
          (goto-char (point-min))
          (forward-line (1- import))
          (beginning-of-line)
          (if arg
              (comment-region (line-beginning-position) (line-end-position))
            (let ((kill-whole-line t))
              (kill-line))))
        (message "Removed %d imports" (length lines)))
      (if flymake-state (flymake-mode-on)))))

(provide 'go-mode)
