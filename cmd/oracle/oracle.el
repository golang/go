;;;
;;; Integration of the Go 'oracle' analysis tool into Emacs.
;;;
;;; To install the Go oracle, run:
;;; % export GOROOT=... GOPATH=...
;;; % go get code.google.com/p/go.tools/cmd/oracle
;;; % mv $GOPATH/bin/oracle $GOROOT/bin/
;;;
;;; Load this file into Emacs and set go-oracle-scope to your
;;; configuration. Then, find a file of Go source code, enable
;;; go-oracle-mode, select an expression of interest, and press `C-c C-o d'
;;; (for "describe") or run one of the other go-oracle-xxx commands.
;;;
;;; TODO(adonovan): simplify installation and configuration by making
;;; oracle a subcommand of 'go tool'.

(require 'compile)
(require 'go-mode)
(require 'cl)

(defgroup go-oracle nil
  "Options specific to the Go oracle."
  :group 'go)

(defcustom go-oracle-command (concat (car (go-root-and-paths)) "/bin/oracle")
  "The Go oracle command; the default is $GOROOT/bin/oracle."
  :type 'string
  :group 'go-oracle)

(defcustom go-oracle-scope ""
  "The scope of the analysis.  See `go-oracle-set-scope'."
  :type 'string
  :group 'go-oracle)

(defvar go-oracle--scope-history
  nil
  "History of values supplied to `go-oracle-set-scope'.")

;; TODO(adonovan): I'd like to get rid of this separate mode since it
;; makes it harder to use the oracle.
(defvar go-oracle-mode-map
  (let ((m (make-sparse-keymap)))
    (define-key m (kbd "C-c C-o t") #'go-oracle-describe) ; t for type
    (define-key m (kbd "C-c C-o f") #'go-oracle-freevars)
    (define-key m (kbd "C-c C-o g") #'go-oracle-callgraph)
    (define-key m (kbd "C-c C-o i") #'go-oracle-implements)
    (define-key m (kbd "C-c C-o c") #'go-oracle-peers)  ; c for channel
    (define-key m (kbd "C-c C-o r") #'go-oracle-referrers)
    (define-key m (kbd "C-c C-o d") #'go-oracle-definition)
    (define-key m (kbd "C-c C-o p") #'go-oracle-pointsto)
    (define-key m (kbd "C-c C-o s") #'go-oracle-callstack)
    (define-key m (kbd "C-c C-o <") #'go-oracle-callers)
    (define-key m (kbd "C-c C-o >") #'go-oracle-callees)
    m))

;; TODO(dominikh): Rethink set-scope some. Setting it to a file is
;; painful because it doesn't use find-file, and variables/~ aren't
;; expanded. Setting it to an import path is somewhat painful because
;; it doesn't make use of go-mode's import path completion. One option
;; would be having two different functions, but then we can't
;; automatically call it when no scope has been set. Also it wouldn't
;; easily allow specifying more than one file/package.
(defun go-oracle-set-scope ()
  "Set the scope for the Go oracle, prompting the user to edit the
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
  (let ((scope (read-from-minibuffer "Go oracle scope: "
                                     go-oracle-scope
                                     nil
                                     nil
                                     'go-oracle--scope-history)))
    (if (string-equal "" scope)
        (error "You must specify a non-empty scope for the Go oracle"))
    (setq go-oracle-scope scope)))

(defun go-oracle--run (mode)
  "Run the Go oracle in the specified MODE, passing it the
selected region of the current buffer.  Process the output to
replace each file name with a small hyperlink.  Display the
result."
  (if (not buffer-file-name)
      (error "Cannot use oracle on a buffer without a file name"))
  ;; It's not sufficient to save a modified buffer since if
  ;; gofmt-before-save is on the before-save-hook, saving will
  ;; disturb the selected region.
  (if (buffer-modified-p)
      (error "Please save the buffer before invoking go-oracle"))
  (if (string-equal "" go-oracle-scope)
      (go-oracle-set-scope))
  (let* ((filename (file-truename buffer-file-name))
         (posflag (if (use-region-p)
                      (format "-pos=%s:#%d,#%d"
                              filename
                              (1- (go--position-bytes (region-beginning)))
                              (1- (go--position-bytes (region-end))))
                    (format "-pos=%s:#%d"
                            filename
                            (1- (position-bytes (point))))))
         ;; This would be simpler if we could just run 'go tool oracle'.
         (env-vars (go-root-and-paths))
         (goroot-env (concat "GOROOT=" (car env-vars)))
         (gopath-env (concat "GOPATH=" (mapconcat #'identity (cdr env-vars) ":"))))
    (with-current-buffer (get-buffer-create "*go-oracle*")
      (setq buffer-read-only nil)
      (erase-buffer)
      (insert "Go Oracle\n")
      (let ((args (append (list go-oracle-command nil t nil posflag mode)
                          (split-string go-oracle-scope " " t))))
        ;; Log the command to *Messages*, for debugging.
        (message "Command: %s:" args)
        (message nil) ; clears/shrinks minibuffer

        (message "Running oracle...")
        ;; Use dynamic binding to modify/restore the environment
        (let ((process-environment (list* goroot-env gopath-env process-environment)))
            (apply #'call-process args)))
      (insert "\n")
      (compilation-mode)
      (setq compilation-error-screen-columns nil)

      ;; Hide the file/line info to save space.
      ;; Replace each with a little widget.
      ;; compilation-mode + this loop = slooow.
      ;; TODO(adonovan): have oracle give us JSON
      ;; and we'll do the markup directly.
      (let ((buffer-read-only nil)
            (p 1))
        (while (not (null p))
          (let ((np (compilation-next-single-property-change p 'compilation-message)))
            ;; TODO(adonovan): this can be verbose in the *Messages* buffer.
            ;; (message "Post-processing link (%d%%)" (/ (* p 100) (point-max)))
            (if np
                (when (equal (line-number-at-pos p) (line-number-at-pos np))
                  ;; np is (typically) the space following ":"; consume it too.
                  (put-text-property p np 'display "▶")
                  (goto-char np)
                  (insert " ")))
            (setq p np)))
        (message nil))

      (let ((w (display-buffer (current-buffer))))
        (balance-windows)
        (shrink-window-if-larger-than-buffer w)
        (set-window-point w (point-min))))))

(defun go-oracle-callees ()
  "Show possible callees of the function call at the current point."
  (interactive)
  (go-oracle--run "callees"))

(defun go-oracle-callers ()
  "Show the set of callers of the function containing the current point."
  (interactive)
  (go-oracle--run "callers"))

(defun go-oracle-callgraph ()
  "Show the callgraph of the current program."
  (interactive)
  (go-oracle--run "callgraph"))

(defun go-oracle-callstack ()
  "Show an arbitrary path from a root of the call graph to the
function containing the current point."
  (interactive)
  (go-oracle--run "callstack"))

(defun go-oracle-definition ()
  "Show the definition of the selected identifier."
  (interactive)
  (go-oracle--run "definition"))

(defun go-oracle-describe ()
  "Describe the selected syntax, its kind, type and methods."
  (interactive)
  (go-oracle--run "describe"))

(defun go-oracle-pointsto ()
  "Show what the selected expression points to."
  (interactive)
  (go-oracle--run "pointsto"))

(defun go-oracle-implements ()
  "Describe the 'implements' relation for types in the package
containing the current point."
  (interactive)
  (go-oracle--run "implements"))

(defun go-oracle-freevars ()
  "Enumerate the free variables of the current selection."
  (interactive)
  (go-oracle--run "freevars"))

(defun go-oracle-peers ()
  "Enumerate the set of possible corresponding sends/receives for
this channel receive/send operation."
  (interactive)
  (go-oracle--run "peers"))

(defun go-oracle-referrers ()
  "Enumerate all references to the object denoted by the selected
identifier."
  (interactive)
  (go-oracle--run "referrers"))

;; TODO(dominikh): better docstring
(define-minor-mode go-oracle-mode "Oracle minor mode for go-mode

Keys specific to go-oracle-mode:
\\{go-oracle-mode-map}"
  nil " oracle" go-oracle-mode-map)

(provide 'go-oracle)
