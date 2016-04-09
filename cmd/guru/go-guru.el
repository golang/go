;;; go-guru.el --- Integration of the Go 'guru' analysis tool into Emacs.

;; Copyright 2016 The Go Authors. All rights reserved.
;; Use of this source code is governed by a BSD-style
;; license that can be found in the LICENSE file.

;; Version: 0.1
;; Package-Requires: ((go-mode "1.3.1") (cl-lib "0.5"))
;; Keywords: tools

;;; Commentary:

;; To install the Go guru, run:
;;
;; $ go get golang.org/x/tools/cmd/guru
;;
;; Load this file into Emacs and set go-guru-scope to your
;; configuration.  Then, find a file of Go source code,
;; select an expression of interest, and press `C-c C-o d' (for "describe")
;; or run one of the other go-guru-xxx commands.

;;; Code:

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

(defcustom go-guru-build-tags ""
  "Build tags passed to guru."
  :type 'string
  :group 'go-guru)

(defface go-guru-hl-identifier-face
  '((t (:inherit highlight)))
  "Face used for highlighting identifiers in `go-guru-hl-identifier'."
  :group 'go-guru)

(defcustom go-guru-debug nil
  "Print debug messages when running guru."
  :type 'boolean
  :group 'go-guru)

(defcustom go-guru-hl-identifier-idle-time 0.5
  "How long to wait after user input before highlighting the current identifier."
  :type 'float
  :group 'go-guru)

(defvar go-guru--current-hl-identifier-idle-time
  0
  "The current delay for hl-identifier-mode.")

(defvar go-guru--hl-identifier-timer
  nil
  "The global timer used for highlighting identifiers.")

(defvar go-guru--last-enclosing
  nil
  "The remaining enclosing regions of the previous go-expand-region invocation.")

;; Extend go-mode-map.
(let ((m (define-prefix-command 'go-guru-map)))
  (define-key m "d" #'go-guru-describe)
  (define-key m "f" #'go-guru-freevars)
  (define-key m "i" #'go-guru-implements)
  (define-key m "c" #'go-guru-peers)  ; c for channel
  (define-key m "r" #'go-guru-referrers)
  (define-key m "j" #'go-guru-definition) ; j for jump
  (define-key m "p" #'go-guru-pointsto)
  (define-key m "s" #'go-guru-callstack) ; s for stack
  (define-key m "e" #'go-guru-whicherrs) ; e for error
  (define-key m "<" #'go-guru-callers)
  (define-key m ">" #'go-guru-callees)
  (define-key m "x" #'go-guru-expand-region)) ;; x for expand

(define-key go-mode-map (kbd "C-c C-o") #'go-guru-map)

;;;###autoload
(defun go-guru-set-scope ()
  "Set the scope for the Go guru, prompting the user to edit the previous scope.

The scope restricts analysis to the specified packages.
Its value is a comma-separated list of patterns of these forms:
	golang.org/x/tools/cmd/guru     # a single package
	golang.org/x/tools/...          # all packages beneath dir
	...                             # the entire workspace.

A pattern preceded by '-' is negative, so the scope
	encoding/...,-encoding/xml
matches all encoding packages except encoding/xml."
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
if not already set.  Mark up the output using `compilation-mode`,
replacing each file name with a small hyperlink, and display the
result."
  (let ((output (go-guru--exec mode need-scope))
	(display (get-buffer-create "*go-guru*"))
	(dir default-directory))
    (with-current-buffer display
      (setq buffer-read-only nil)
      (setq default-directory dir)
      (erase-buffer)
      (insert-buffer-substring output)
      (go-guru--compilation-markup))))

(defun go-guru--exec (mode &optional need-scope flags allow-unnamed)
  "Execute the Go guru in the specified MODE, passing it the
selected region of the current buffer. If NEED-SCOPE, prompt for
a scope if not already set. If ALLOW-UNNAMED is non-nil, a
synthetic file for the unnamed buffer will be created. This
should only be used with queries that work on single files
only (e.g. 'what'). If ALLOW-UNNAMED is nil and the buffer has no
associated name, an error will be signaled.

Return the output buffer."
  (or
   buffer-file-name
   allow-unnamed
   (error "Cannot use guru on a buffer without a file name"))
  (and need-scope
       (string-equal "" go-guru-scope)
       (go-guru-set-scope))
  (let* ((is-unnamed (not buffer-file-name))
	 (filename (file-truename (or buffer-file-name "synthetic.go")))
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
         (output-buffer (get-buffer-create "*go-guru-output*"))
         (buf (current-buffer)))
    (with-current-buffer output-buffer
      (setq buffer-read-only nil)
      (erase-buffer))
    (with-current-buffer (get-buffer-create "*go-guru-input*")
      (setq buffer-read-only nil)
      (erase-buffer)
      (if is-unnamed
	  (go-guru--insert-modified-file filename buf)
        (go-guru--insert-modified-files))
      (let* ((args (append (list "-modified"
                                 "-scope" go-guru-scope
                                 "-tags" go-guru-build-tags)
			   flags
			   (list mode posn))))
	;; Log the command to *Messages*, for debugging.
 	(when go-guru-debug
	  (message "Command: %s:" args)
	  (message nil) ; clears/shrinks minibuffer
	  (message "Running guru %s..." mode))
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
  (goto-char (point-max))
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
    (set-window-point w (point-min))))

(defun go-guru--insert-modified-files ()
  "Insert the contents of each modified Go buffer into the
current buffer in the format specified by guru's -modified flag."
  (mapc #'(lambda (b)
            (and (buffer-modified-p b)
                 (buffer-file-name b)
                 (string= (file-name-extension (buffer-file-name b)) "go")
                 (go-guru--insert-modified-file (buffer-file-name b) b)))
        (buffer-list)))

(defun go-guru--insert-modified-file (name buffer)
  (insert (format "%s\n%d\n" name (go-guru--buffer-size-bytes buffer)))
  (insert-buffer-substring buffer))

(defun go-guru--buffer-size-bytes (&optional buffer)
  "Return the number of bytes in the current buffer.
If BUFFER, return the number of characters in that buffer instead."
  (with-current-buffer (or buffer (current-buffer))
    (string-bytes (buffer-substring (point-min)
                                    (point-max)))))

;; FIXME(dominikh): go-guru--goto-pos-no-file and go-guru--goto-pos
;; assume that Guru is giving rune offsets in the columns field.
;; However, it is giving us byte offsets, causing us to highlight
;; wrong ranges as soon as there's any multi-byte runes in the line.
(defun go-guru--goto-pos (posn)
  "Find the file containing the position POSN (of the form `file:line:col')
set the point to it, switching the current buffer."
  (let ((file-line-pos (split-string posn ":")))
    (find-file (car file-line-pos))
    (goto-char (point-min))
    (forward-line (1- (string-to-number (cadr file-line-pos))))
    (forward-char (1- (string-to-number (caddr file-line-pos))))))

(defun go-guru--goto-pos-no-file (posn)
  "Given `file:line:col', go to the line and column. The file
component will be ignored."
  (let ((file-line-pos (split-string posn ":")))
    (goto-char (point-min))
    (forward-line (1- (string-to-number (cadr file-line-pos))))
    (forward-char (1- (string-to-number (caddr file-line-pos))))))

;;;###autoload
(defun go-guru-callees ()
  "Show possible callees of the function call at the current point."
  (interactive)
  (go-guru--run "callees" t))

;;;###autoload
(defun go-guru-callers ()
  "Show the set of callers of the function containing the current point."
  (interactive)
  (go-guru--run "callers" t))

;;;###autoload
(defun go-guru-callstack ()
  "Show an arbitrary path from a root of the call graph to the
function containing the current point."
  (interactive)
  (go-guru--run "callstack" t))

;;;###autoload
(defun go-guru-definition ()
  "Jump to the definition of the selected identifier."
  (interactive)
  (let* ((res (with-current-buffer (go-guru--exec "definition" nil '("-json"))
		(goto-char (point-min))
		(json-read)))
	 (desc (cdr (assoc 'desc res))))
    (push-mark)
    (ring-insert find-tag-marker-ring (point-marker))
    (go-guru--goto-pos (cdr (assoc 'objpos res)))
    (message "%s" desc)))

;;;###autoload
(defun go-guru-describe ()
  "Describe the selected syntax, its kind, type and methods."
  (interactive)
  (go-guru--run "describe"))

;;;###autoload
(defun go-guru-pointsto ()
  "Show what the selected expression points to."
  (interactive)
  (go-guru--run "pointsto" t))

;;;###autoload
(defun go-guru-implements ()
  "Describe the 'implements' relation for types in the package
containing the current point."
  (interactive)
  (go-guru--run "implements"))

;;;###autoload
(defun go-guru-freevars ()
  "Enumerate the free variables of the current selection."
  (interactive)
  (go-guru--run "freevars"))

;;;###autoload
(defun go-guru-peers ()
  "Enumerate the set of possible corresponding sends/receives for
this channel receive/send operation."
  (interactive)
  (go-guru--run "peers" t))

;;;###autoload
(defun go-guru-referrers ()
  "Enumerate all references to the object denoted by the selected
identifier."
  (interactive)
  (go-guru--run "referrers"))

;;;###autoload
(defun go-guru-whicherrs ()
  "Show globals, constants and types to which the selected
expression (of type 'error') may refer."
  (interactive)
  (go-guru--run "whicherrs" t))

(defun go-guru-what ()
  "Run a 'what' query and return the parsed JSON response as an
association list."
  (with-current-buffer (go-guru--exec "what" nil '("-json") t)
    (goto-char (point-min))
    (json-read)))

(defun go-guru--hl-symbols (posn face id)
  "Highlight the symbols at the positions POSN by creating
overlays with face FACE. The attribute 'go-guru-overlay on the
overlays will be set to ID."
  (save-excursion
    (mapc (lambda (pos)
	    (go-guru--goto-pos-no-file pos)
	    (let ((x (make-overlay (point) (+ (point) (length (current-word))))))
	      (overlay-put x 'go-guru-overlay id)
	      (overlay-put x 'face face)))
	  posn)))

;;;###autoload
(defun go-guru-unhighlight-identifiers ()
  "Remove highlights from previously highlighted identifier."
  (remove-overlays nil nil 'go-guru-overlay 'sameid))

;;;###autoload
(defun go-guru-hl-identifier ()
  "Highlight all instances of the identifier under point. Removes
highlights from previously highlighted identifier."
  (interactive)
  (go-guru-unhighlight-identifiers)
  (go-guru--hl-identifier))

(defun go-guru--hl-identifier ()
  "Highlight all instances of the identifier under point."
  (let ((posn (cdr (assoc 'sameids (go-guru-what)))))
    (go-guru--hl-symbols posn 'go-guru-hl-identifier-face 'sameid)))

(defun go-guru--hl-identifiers-function ()
  "Function run after an idle timeout, highlighting the
identifier at point, if necessary."
  (when go-guru-hl-identifier-mode
    (unless (go-guru--on-overlay-p 'sameid)
      ;; Ignore guru errors. Otherwise, we might end up with an error
      ;; every time the timer runs, e.g. because of a malformed
      ;; buffer.
      (condition-case nil
          (go-guru-hl-identifier)
        (error nil)))
    (unless (eq go-guru--current-hl-identifier-idle-time go-guru-hl-identifier-idle-time)
      (go-guru--hl-set-timer))))

(defun go-guru--hl-set-timer ()
  (if go-guru--hl-identifier-timer
      (cancel-timer go-guru--hl-identifier-timer))
  (setq go-guru--current-hl-identifier-idle-time go-guru-hl-identifier-idle-time)
  (setq go-guru--hl-identifier-timer (run-with-idle-timer
				      go-guru-hl-identifier-idle-time
				      t
				      #'go-guru--hl-identifiers-function)))

;;;###autoload
(define-minor-mode go-guru-hl-identifier-mode
  "Highlight instances of the identifier at point after a short
timeout."
  :group 'go-guru
  (if go-guru-hl-identifier-mode
      (progn
	(go-guru--hl-set-timer)
	;; Unhighlight if point moves off identifier
	(add-hook 'post-command-hook #'go-guru--hl-identifiers-post-command-hook nil t)
	;; Unhighlight any time the buffer changes
	(add-hook 'before-change-functions #'go-guru--hl-identifiers-before-change-function nil t))
    (remove-hook 'post-command-hook #'go-guru--hl-identifiers-post-command-hook t)
    (remove-hook 'before-change-functions #'go-guru--hl-identifiers-before-change-function t)
    (go-guru-unhighlight-identifiers)))

(defun go-guru--on-overlay-p (id)
  "Return whether point is on a guru overlay of type ID."
  (find-if (lambda (el) (eq (overlay-get el 'go-guru-overlay) id)) (overlays-at (point))))

(defun go-guru--hl-identifiers-post-command-hook ()
  (if (and go-guru-hl-identifier-mode
	   (not (go-guru--on-overlay-p 'sameid)))
      (go-guru-unhighlight-identifiers)))

(defun go-guru--hl-identifiers-before-change-function (_beg _end)
  (go-guru-unhighlight-identifiers))

;; TODO(dominikh): a future feature may be to cycle through all uses
;; of an identifier.

(defun go-guru--enclosing ()
  "Return a list of enclosing regions, with duplicates removed."
  (let ((enclosing (cdr (assoc 'enclosing (go-guru-what)))))
    (cl-remove-duplicates enclosing
                          :test (lambda (a b)
                                  (and (= (cdr (assoc 'start a))
                                          (cdr (assoc 'start b)))
                                       (= (cdr (assoc 'end a))
                                          (cdr (assoc 'end b))))))))

(defun go-guru-expand-region ()
  "Expand region to the next enclosing syntactic unit."
  (interactive)
  (let* ((enclosing (if (eq last-command #'go-guru-expand-region)
                        go-guru--last-enclosing
                      (go-guru--enclosing)))
         (block (if (> (length enclosing) 0) (elt enclosing 0))))
    (when block
      (goto-char (1+ (cdr (assoc 'start block))))
      (set-mark (1+ (cdr (assoc 'end block))))
      (setq go-guru--last-enclosing (subseq enclosing 1))
      (message "Region: %s" (cdr (assoc 'desc block)))
      (setq deactivate-mark nil))))


(provide 'go-guru)

;;; go-guru.el ends here
