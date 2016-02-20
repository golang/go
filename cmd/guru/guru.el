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
(require 'json)
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
(let ((m go-guru-map))
  (define-key m "d" #'go-guru-describe)
  (define-key m "f" #'go-guru-freevars)
  (define-key m "i" #'go-guru-implements)
  (define-key m "c" #'go-guru-peers)  ; c for channel
  (define-key m "r" #'go-guru-referrers)
  (define-key m "j" #'go-guru-definition) ; j for jump
  (define-key m "p" #'go-guru-pointsto)
  (define-key m "s" #'go-guru-callstack) ; s for stack
  (define-key m "<" #'go-guru-callers)
  (define-key m ">" #'go-guru-callees))

(define-key go-mode-map (kbd "C-c C-o") #'go-guru-map)

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
1) a set of packages whose main functions will be analyzed.
2) a list of *.go filenames; they will treated like as a single
   package (see #3).
3) a single package whose main function and/or Test* functions
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
  "Run the Go guru in the specified MODE, passing it the selected
region of the current buffer.  If NEED-SCOPE, prompt for a scope
if not already set.  Mark up the output using `compilation-node`,
replacing each file name with a small hyperlink, and display the
result."
  (with-current-buffer (go-guru--exec mode need-scope)
    (go-guru--compilation-markup)))

(defun go-guru--exec (mode &optional need-scope flags)
  "Execute the Go guru in the specified MODE, passing it the
selected region of the current buffer.  If NEED-SCOPE, prompt for
a scope if not already set.  Return the output buffer."
  (if (not buffer-file-name)
      (error "Cannot use guru on a buffer without a file name"))
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
         (gopath-env (concat "GOPATH=" (mapconcat #'identity (cdr env-vars) ":")))
	 (output-buffer (get-buffer-create "*go-guru*")))
    (with-current-buffer output-buffer
      (setq buffer-read-only nil)
      (erase-buffer))
    (with-current-buffer (get-buffer-create "*go-guru-input*")
      (setq buffer-read-only nil)
      (erase-buffer)
      (go-guru--insert-modified-files)
      (let* ((args (append (list "-modified"
				 "-scope" go-guru-scope)
			   flags
			   (list mode posn))))
	;; Log the command to *Messages*, for debugging.
	(message "Command: %s:" args)
	(message nil) ; clears/shrinks minibuffer
	(message "Running guru %s..." mode)
	;; Use dynamic binding to modify/restore the environment
	(let* ((process-environment (list* goroot-env gopath-env process-environment))
	       (c-p-args (append (list (point-min)
				       (point-max)
				       go-guru-command
				       nil ; delete
				       output-buffer
				       t)
				 args))
	       (exitcode (apply #'call-process-region c-p-args)))
	  ;; If the command fails, don't show the output buffer,
	  ;; but use its contents (sans final \n) as an error.
	  (unless (zerop exitcode)
	    (with-current-buffer output-buffer
	      (bury-buffer)
	      (error "%s" (buffer-substring (point-min) (1- (point-max)))))))))
    output-buffer))

(defun go-guru--compilation-markup ()
  "Present guru output in the current buffer using `compilation-mode'."
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
    (shrink-window-if-larger-than-buffer w)
    (set-window-point w (point-min))))

(defun go-guru--insert-modified-files ()
  "Insert the contents of each modified Go buffer into the
current buffer in the format specified by guru's -modified flag."
  (mapc #'(lambda (b)
	    (and (buffer-modified-p b)
		 (buffer-file-name b)
		 (string= (file-name-extension (buffer-file-name b)) "go")
		 (progn
		   (insert (format "%s\n%d\n"
				   (buffer-file-name b)
                                   (go-guru--buffer-size-bytes b)))
                   (insert-buffer-substring b))))
	(buffer-list)))

(defun go-guru--buffer-size-bytes (&optional buffer)
  "Return the number of bytes in the current buffer.
If BUFFER, return the number of characters in that buffer instead."
  (with-current-buffer (or buffer (current-buffer))
    (string-bytes (buffer-substring (point-min)
                                    (point-max)))))


(defun go-guru--goto-pos (posn)
  "Find the file containing the position POSN (of the form `file:line:col')
set the point to it, switching the current buffer."
  (let ((file-line-pos (split-string posn ":")))
    (find-file (car file-line-pos))
    (goto-char (point-min))
    ;; NB: go/token's column offsets are byte- not rune-based.
    (forward-line (1- (string-to-number (cadr file-line-pos))))
    (forward-char (1- (string-to-number (caddr file-line-pos))))))

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
  "Jump to the definition of the selected identifier."
  (interactive)
  ;; TODO(adonovan): use -format=sexpr when available to avoid a
  ;; dependency and to simplify parsing.
  (let* ((res (with-current-buffer (go-guru--exec "definition" nil '("-format=json"))
		(goto-char (point-min))
		(cdr (car (json-read)))))
	 (desc (cdr (assoc 'desc res))))
    (go-guru--goto-pos (cdr (assoc 'objpos res)))
    (message "%s" desc)))

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
