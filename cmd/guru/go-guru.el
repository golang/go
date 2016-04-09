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

(defun go-guru--set-scope-if-empty ()
  (if (string-equal "" go-guru-scope)
      (go-guru-set-scope)))

(defun go-guru--json (mode)
  "Execute the Go guru in the specified MODE, passing it the
selected region of the current buffer, requesting JSON output.
Parse and return the resulting JSON object."
  ;; A "what" query works even in a buffer without a file name.
  (let* ((filename (file-truename (or buffer-file-name "synthetic.go")))
	 (cmd (go-guru--command mode filename '("-json")))
	 (buf (current-buffer))
	 ;; Use temporary buffers to avoid conflict with go-guru--start.
	 (json-buffer (generate-new-buffer "*go-guru-json-output*"))
	 (input-buffer (generate-new-buffer "*go-guru-json-input*")))
    (unwind-protect
	;; Run guru, feeding it the input buffer (modified files).
	(with-current-buffer input-buffer
	  (go-guru--insert-modified-files)
	  (unless (buffer-file-name buf)
	    (go-guru--insert-modified-file filename buf))
	  (let ((exitcode (apply #'call-process-region
				 (append (list (point-min)
					       (point-max)
					       (car cmd) ; guru
					       nil ; delete
					       json-buffer ; output
					       nil) ; display
					 (cdr cmd))))) ; args
	    (with-current-buffer json-buffer
	      (unless (zerop exitcode)
		;; Failed: use buffer contents (sans final \n) as an error.
		(error "%s" (buffer-substring (point-min) (1- (point-max)))))
	      ;; Success: parse JSON.
	      (goto-char (point-min))
	      (json-read))))
      ;; Clean up temporary buffers.
      (kill-buffer json-buffer)
      (kill-buffer input-buffer))))

(define-compilation-mode go-guru-output-mode "Go guru"
  "Go guru output mode is a variant of `compilation-mode' for the
output of the Go guru tool."
  (set (make-local-variable 'compilation-error-screen-columns) nil)
  (set (make-local-variable 'compilation-filter-hook) #'go-guru--compilation-filter-hook)
  (set (make-local-variable 'compilation-start-hook) #'go-guru--compilation-start-hook))

(defun go-guru--compilation-filter-hook ()
  "Post-process a blob of input to the go-guru-output buffer."
  ;; For readability, truncate each "file:line:col:" prefix to a fixed width.
  ;; If the prefix is longer than 20, show "…/last/19chars.go".
  ;; This usually includes the last segment of the package name.
  ;; Hide the line and column numbers.
  (let ((start compilation-filter-start)
	(end (point)))
    (goto-char start)
    (unless (bolp)
      ;; TODO(adonovan): not quite right: the filter may be called
      ;; with chunks of output containing incomplete lines.  Moving to
      ;; beginning-of-line may cause duplicate post-processing.
      (beginning-of-line))
    (setq start (point))
    (while (< start end)
      (let ((p (search-forward ": " end t)))
	(if (null p)
	    (setq start end) ; break out of loop
	  (setq p (1- p)) ; exclude final space
	  (let* ((posn (buffer-substring-no-properties start p))
		 (flen (search ":" posn)) ; length of filename
		 (filename (if (< flen 19)
			       (substring posn 0 flen)
			     (concat "…" (substring posn (- flen 19) flen)))))
	    (put-text-property start p 'display filename)
	    (forward-line 1)
	    (setq start (point))))))))

(defun go-guru--compilation-start-hook (proc)
  "Erase default output header inserted by `compilation-mode'."
  (with-current-buffer (process-buffer proc)
    (let ((inhibit-read-only t))
      (beginning-of-buffer)
      (delete-region (point) (point-max)))))

(defun go-guru--start (mode)
  "Start an asynchronous Go guru process for the specified query
MODE, passing it the selected region of the current buffer, and
feeding its standard input with the contents of all modified Go
buffers.  Its output is handled by `go-guru-output-mode', a
variant of `compilation-mode'."
  (or buffer-file-name
      (error "Cannot use guru on a buffer without a file name"))
  (let* ((filename (file-truename buffer-file-name))
	 (cmd (combine-and-quote-strings (go-guru--command mode filename)))
	 (process-connection-type nil) ; use pipe (not pty) so EOF closes stdin
	 (procbuf (compilation-start cmd 'go-guru-output-mode)))
    (with-current-buffer procbuf
      (setq truncate-lines t)) ; the output is neater without line wrapping
    (with-current-buffer (get-buffer-create "*go-guru-input*")
      (erase-buffer)
      (go-guru--insert-modified-files)
      (process-send-region procbuf (point-min) (point-max))
      (process-send-eof procbuf))
    procbuf))

(defun go-guru--command (mode filename &optional flags)
  "Return a command and argument list for a Go guru query of MODE, passing it
the selected region of the current buffer.  FILENAME is the
effective name of the current buffer."
  (let* ((posn (if (use-region-p)
		   (format "%s:#%d,#%d"
			   filename
			   (1- (go--position-bytes (region-beginning)))
			   (1- (go--position-bytes (region-end))))
		 (format "%s:#%d"
			 filename
			 (1- (go--position-bytes (point))))))
	 (cmd (append (list go-guru-command
			    "-modified"
			    "-scope" go-guru-scope
			    (format "-tags=%s" (mapconcat 'identity go-guru-build-tags ",")))
		      flags
		      (list mode
			    posn))))
    ;; Log the command to *Messages*, for debugging.
    (when go-guru-debug
      (message "go-guru--command: %s" cmd)
      (message nil)) ; clear/shrink minibuffer
    cmd))

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

(defun go-guru--goto-byte (offset)
  "Go to the OFFSETth byte in the buffer."
  (goto-char (byte-to-position offset)))

(defun go-guru--goto-byte-column (offset)
  "Go to the OFFSETth byte in the current line."
  (goto-char (byte-to-position (+ (position-bytes (point-at-bol)) (1- offset)))))

(defun go-guru--goto-pos (posn)
  "Find the file containing the position POSN (of the form `file:line:col')
set the point to it, switching the current buffer."
  (let ((file-line-pos (split-string posn ":")))
    (find-file (car file-line-pos))
    (goto-char (point-min))
    (forward-line (1- (string-to-number (cadr file-line-pos))))
    (go-guru--goto-byte-column (string-to-number (caddr file-line-pos)))))

(defun go-guru--goto-pos-no-file (posn)
  "Given `file:line:col', go to the line and column. The file
component will be ignored."
  (let ((file-line-pos (split-string posn ":")))
    (goto-char (point-min))
    (forward-line (1- (string-to-number (cadr file-line-pos))))
    (go-guru--goto-byte-column (string-to-number (caddr file-line-pos)))))

;;;###autoload
(defun go-guru-callees ()
  "Show possible callees of the function call at the current point."
  (interactive)
  (go-guru--set-scope-if-empty)
  (go-guru--start "callees"))

;;;###autoload
(defun go-guru-callers ()
  "Show the set of callers of the function containing the current point."
  (interactive)
  (go-guru--set-scope-if-empty)
  (go-guru--start "callers"))

;;;###autoload
(defun go-guru-callstack ()
  "Show an arbitrary path from a root of the call graph to the
function containing the current point."
  (interactive)
  (go-guru--set-scope-if-empty)
  (go-guru--start "callstack"))

;;;###autoload
(defun go-guru-definition ()
  "Jump to the definition of the selected identifier."
  (interactive)
  (or buffer-file-name
      (error "Cannot use guru on a buffer without a file name"))
  (let* ((res (go-guru--json "definition"))
	 (desc (cdr (assoc 'desc res))))
    (push-mark)
    (ring-insert find-tag-marker-ring (point-marker))
    (go-guru--goto-pos (cdr (assoc 'objpos res)))
    (message "%s" desc)))

;;;###autoload
(defun go-guru-describe ()
  "Describe the selected syntax, its kind, type and methods."
  (interactive)
  (go-guru--start "describe"))

;;;###autoload
(defun go-guru-pointsto ()
  "Show what the selected expression points to."
  (interactive)
  (go-guru--set-scope-if-empty)
  (go-guru--start "pointsto"))

;;;###autoload
(defun go-guru-implements ()
  "Describe the 'implements' relation for types in the package
containing the current point."
  (interactive)
  (go-guru--start "implements"))

;;;###autoload
(defun go-guru-freevars ()
  "Enumerate the free variables of the current selection."
  (interactive)
  (go-guru--start "freevars"))

;;;###autoload
(defun go-guru-peers ()
  "Enumerate the set of possible corresponding sends/receives for
this channel receive/send operation."
  (interactive)
  (go-guru--set-scope-if-empty)
  (go-guru--start "peers"))

;;;###autoload
(defun go-guru-referrers ()
  "Enumerate all references to the object denoted by the selected
identifier."
  (interactive)
  (go-guru--start "referrers"))

;;;###autoload
(defun go-guru-whicherrs ()
  "Show globals, constants and types to which the selected
expression (of type 'error') may refer."
  (interactive)
  (go-guru--set-scope-if-empty)
  (go-guru--start "whicherrs"))

(defun go-guru-what ()
  "Run a 'what' query and return the parsed JSON response as an
association list."
  (go-guru--json "what"))

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
  "Return a list of enclosing regions."
  (cdr (assoc 'enclosing (go-guru-what))))

(defun go-guru--enclosing-unique ()
  "Return a list of enclosing regions, with duplicates removed.
Two regions are considered equal if they have the same start and
end point."
  (let ((enclosing (go-guru--enclosing)))
    (cl-remove-duplicates enclosing
			  :from-end t
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
		      (go-guru--enclosing-unique)))
	 (block (if (> (length enclosing) 0) (elt enclosing 0))))
    (when block
      (go-guru--goto-byte (1+ (cdr (assoc 'start block))))
      (set-mark (byte-to-position (1+ (cdr (assoc 'end block)))))
      (setq go-guru--last-enclosing (subseq enclosing 1))
      (message "Region: %s" (cdr (assoc 'desc block)))
      (setq deactivate-mark nil))))


(provide 'go-guru)

;; Local variables:
;; indent-tabs-mode: t
;; tab-width: 8
;; End

;;; go-guru.el ends here
