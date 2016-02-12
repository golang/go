;;;
;;; Integration of the Go 'guru' analysis tool into Emacs.
;;;
;;; To install the Go guru, run:
;;; $ go get golang.org/x/tools/cmd/guru
;;;
;;; Load this file into Emacs and set go-guru-scope to your
;;; configuration. Then, find a file of Go source code, enable
;;; go-guru-mode, select an expression of interest, and press `C-c C-o d'
;;; (for "describe") or run one of the other go-guru-xxx commands.

(require 'compile)
(require 'go-mode)
(require 'simple)
(require 'cl)

(defgroup go-guru nil
  "Options specific to the Go guru."
  :group 'go)

(defcustom go-guru-command "guru"
  "The Go guru command."
  :type 'string
  :group 'go-guru)

(defcustom go-guru-scope ""
  "The scope of the analysis.  See `go-guru-set-scope'."
  :type 'string
  :group 'go-guru)

(defvar go-guru--scope-history
  nil
  "History of values supplied to `go-guru-set-scope'.")

;; Extend go-mode-map.
(let ((m go-mode-map))
  (define-key m (kbd "C-c C-o t") #'go-guru-describe) ; t for type
  (define-key m (kbd "C-c C-o f") #'go-guru-freevars)
  (define-key m (kbd "C-c C-o g") #'go-guru-callgraph)
  (define-key m (kbd "C-c C-o i") #'go-guru-implements)
  (define-key m (kbd "C-c C-o c") #'go-guru-peers)  ; c for channel
  (define-key m (kbd "C-c C-o r") #'go-guru-referrers)
  (define-key m (kbd "C-c C-o d") #'go-guru-definition)
  (define-key m (kbd "C-c C-o p") #'go-guru-pointsto)
  (define-key m (kbd "C-c C-o s") #'go-guru-callstack)
  (define-key m (kbd "C-c C-o <") #'go-guru-callers)
  (define-key m (kbd "C-c C-o >") #'go-guru-callees)
  (define-key m (kbd "<f5>") #'go-guru-describe)
  (define-key m (kbd "<f6>") #'go-guru-referrers))

;; TODO(dominikh): Rethink set-scope some. Setting it to a file is
;; painful because it doesn't use find-file, and variables/~ aren't
;; expanded. Setting it to an import path is somewhat painful because
;; it doesn't make use of go-mode's import path completion. One option
;; would be having two different functions, but then we can't
;; automatically call it when no scope has been set. Also it wouldn't
;; easily allow specifying more than one file/package.
(defun go-guru-set-scope ()
  "Set the scope for the Go guru, prompting the user to edit the
previous scope.

The scope specifies a set of arguments, separated by spaces.
It may be:
1) a set of packages whose main() functions will be analyzed.
2) a list of *.go filenames; they will treated like as a single
   package (see #3).
3) a single package whose main() function and/or Test* functions
   will be analyzed.

In the common case, this is similar to the argument(s) you would
specify to 'go build'."
  (interactive)
  (let ((scope (read-from-minibuffer "Go guru scope: "
                                     go-guru-scope
                                     nil
                                     nil
                                     'go-guru--scope-history)))
    (if (string-equal "" scope)
        (error "You must specify a non-empty scope for the Go guru"))
    (setq go-guru-scope scope)))

(defun go-guru--run (mode &optional need-scope)
  "Run the Go guru in the specified MODE, passing it the
selected region of the current buffer.  If NEED-SCOPE, prompt for
a scope if not already set.  Process the output to replace each
file name with a small hyperlink.  Display the result."
  (if (not buffer-file-name)
      (error "Cannot use guru on a buffer without a file name"))
  ;; It's not sufficient to save a modified buffer since if
  ;; gofmt-before-save is on the before-save-hook, saving will
  ;; disturb the selected region.
  (if (buffer-modified-p)
      (error "Please save the buffer before invoking go-guru"))
  (and need-scope
       (string-equal "" go-guru-scope)
       (go-guru-set-scope))
  (let* ((filename (file-truename buffer-file-name))
         (posn (if (use-region-p)
		   (format "%s:#%d,#%d"
			   filename
			   (1- (go--position-bytes (region-beginning)))
			   (1- (go--position-bytes (region-end))))
		 (format "%s:#%d"
			 filename
			 (1- (position-bytes (point))))))
         (env-vars (go-root-and-paths))
         (goroot-env (concat "GOROOT=" (car env-vars)))
         (gopath-env (concat "GOPATH=" (mapconcat #'identity (cdr env-vars) ":"))))
    (with-current-buffer (get-buffer-create "*go-guru*")
      (setq buffer-read-only nil)
      (erase-buffer)
      (insert "Go Guru\n")
      (let ((args (list go-guru-command nil t nil
			"-scope" go-guru-scope mode posn)))
        ;; Log the command to *Messages*, for debugging.
        (message "Command: %s:" args)
        (message nil) ; clears/shrinks minibuffer

        (message "Running guru...")
        ;; Use dynamic binding to modify/restore the environment
        (let ((process-environment (list* goroot-env gopath-env process-environment)))
            (apply #'call-process args)))
      (insert "\n")
      (compilation-mode)
      (setq compilation-error-screen-columns nil)

      ;; Hide the file/line info to save space.
      ;; Replace each with a little widget.
      ;; compilation-mode + this loop = slooow.
      ;; TODO(adonovan): have guru give us JSON
      ;; and we'll do the markup directly.
      (let ((buffer-read-only nil)
            (p 1))
        (while (not (null p))
          (let ((np (compilation-next-single-property-change p 'compilation-message)))
            (if np
                (when (equal (line-number-at-pos p) (line-number-at-pos np))
                  ;; Using a fixed width greatly improves readability, so
                  ;; if the filename is longer than 20, show ".../last/17chars.go".
                  ;; This usually includes the last segment of the package name.
                  ;; Don't show the line or column number.
                  (let* ((loc (buffer-substring p np)) ; "/home/foo/go/pkg/file.go:1:2-3:4"
                         (i (search ":" loc)))
                    (setq loc (cond
                               ((null i)  "...")
                               ((>= i 17) (concat "..." (substring loc (- i 17) i)))
                               (t         (substring loc 0 i))))
                    ;; np is (typically) the space following ":"; consume it too.
                    (put-text-property p np 'display (concat loc ":")))
                  (goto-char np)
                  (insert " ")
                  (incf np))) ; so we don't get stuck (e.g. on a panic stack dump)
            (setq p np)))
        (message nil))

      (let ((w (display-buffer (current-buffer))))
        (balance-windows)
        (shrink-window-if-larger-than-buffer w)
        (set-window-point w (point-min))))))

(defun go-guru-callees ()
  "Show possible callees of the function call at the current point."
  (interactive)
  (go-guru--run "callees" t))

(defun go-guru-callers ()
  "Show the set of callers of the function containing the current point."
  (interactive)
  (go-guru--run "callers" t))

(defun go-guru-callgraph ()
  "Show the callgraph of the current program."
  (interactive)
  (go-guru--run "callgraph" t))

(defun go-guru-callstack ()
  "Show an arbitrary path from a root of the call graph to the
function containing the current point."
  (interactive)
  (go-guru--run "callstack" t))

(defun go-guru-definition ()
  "Show the definition of the selected identifier."
  (interactive)
  (go-guru--run "definition"))

(defun go-guru-describe ()
  "Describe the selected syntax, its kind, type and methods."
  (interactive)
  (go-guru--run "describe"))

(defun go-guru-pointsto ()
  "Show what the selected expression points to."
  (interactive)
  (go-guru--run "pointsto" t))

(defun go-guru-implements ()
  "Describe the 'implements' relation for types in the package
containing the current point."
  (interactive)
  (go-guru--run "implements"))

(defun go-guru-freevars ()
  "Enumerate the free variables of the current selection."
  (interactive)
  (go-guru--run "freevars"))

(defun go-guru-peers ()
  "Enumerate the set of possible corresponding sends/receives for
this channel receive/send operation."
  (interactive)
  (go-guru--run "peers" t))

(defun go-guru-referrers ()
  "Enumerate all references to the object denoted by the selected
identifier."
  (interactive)
  (go-guru--run "referrers"))

(defun go-guru-whicherrs ()
  "Show globals, constants and types to which the selected
expression (of type 'error') may refer."
  (interactive)
  (go-guru--run "whicherrs" t))

(provide 'go-guru)
