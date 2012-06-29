;;; go-mode.el --- Major mode for the Go programming language

;;; Commentary:

;; For installation instructions, see go-mode-load.el

;;; To do:

;; * Indentation is *almost* identical to gofmt
;; ** We think struct literal keys are labels and outdent them
;; ** We disagree on the indentation of function literals in arguments
;; ** There are bugs with the close brace of struct literals
;; * Highlight identifiers according to their syntactic context: type,
;;   variable, function call, or tag
;; * Command for adding an import
;; ** Check if it's already there
;; ** Factor/unfactor the import line
;; ** Alphabetize
;; * Remove unused imports
;; ** This is hard, since I have to be aware of shadowing to do it
;;    right
;; * Format region using gofmt

;;; Code:

(eval-when-compile (require 'cl))

(defvar go-mode-syntax-table
  (let ((st (make-syntax-table)))
    ;; Add _ to :word: character class
    (modify-syntax-entry ?_  "w" st)

    ;; Operators (punctuation)
    (modify-syntax-entry ?+  "." st)
    (modify-syntax-entry ?-  "." st)
    (modify-syntax-entry ?*  ". 23" st)                                    ; also part of comments
    (modify-syntax-entry ?/ (if (featurep 'xemacs) ". 1456" ". 124b") st)  ; ditto
    (modify-syntax-entry ?%  "." st)
    (modify-syntax-entry ?&  "." st)
    (modify-syntax-entry ?|  "." st)
    (modify-syntax-entry ?^  "." st)
    (modify-syntax-entry ?!  "." st)
    (modify-syntax-entry ?=  "." st)
    (modify-syntax-entry ?<  "." st)
    (modify-syntax-entry ?>  "." st)

    ;; Strings and comments are font-locked separately.
    (modify-syntax-entry ?\" "." st)
    (modify-syntax-entry ?\' "." st)
    (modify-syntax-entry ?`  "." st)
    (modify-syntax-entry ?\\ "." st)

    ;; Newline is a comment-ender.
    (modify-syntax-entry ?\n "> b" st)

    st)
  "Syntax table for Go mode.")

(defvar go-mode-keywords
  '("break"    "default"     "func"   "interface" "select"
    "case"     "defer"       "go"     "map"       "struct"
    "chan"     "else"        "goto"   "package"   "switch"
    "const"    "fallthrough" "if"     "range"     "type"
    "continue" "for"         "import" "return"    "var")
  "All keywords in the Go language.  Used for font locking and
some syntax analysis.")

(defvar go-mode-font-lock-keywords
  (let ((builtins '("append" "cap" "close" "complex" "copy" "delete" "imag" "len"
                    "make" "new" "panic" "print" "println" "real" "recover"))
        (constants '("nil" "true" "false" "iota"))
        (type-name "\\s *\\(?:[*(]\\s *\\)*\\(?:\\w+\\s *\\.\\s *\\)?\\(\\w+\\)")
        )
    `((go-mode-font-lock-cs-comment 0 font-lock-comment-face t)
      (go-mode-font-lock-cs-string 0 font-lock-string-face t)
      (,(regexp-opt go-mode-keywords 'words) . font-lock-keyword-face)
      (,(regexp-opt builtins 'words) . font-lock-builtin-face)
      (,(regexp-opt constants 'words) . font-lock-constant-face)
      ;; Function names in declarations
      ("\\<func\\>\\s *\\(\\w+\\)" 1 font-lock-function-name-face)
      ;; Function names in methods are handled by function call pattern
      ;; Function names in calls
      ;; XXX Doesn't match if function name is surrounded by parens
      ("\\(\\w+\\)\\s *(" 1 font-lock-function-name-face)
      ;; Type names
      ("\\<type\\>\\s *\\(\\w+\\)" 1 font-lock-type-face)
      (,(concat "\\<type\\>\\s *\\w+\\s *" type-name) 1 font-lock-type-face)
      ;; Arrays/slices/map value type
      ;; XXX Wrong.  Marks 0 in expression "foo[0] * x"
      ;;      (,(concat "]" type-name) 1 font-lock-type-face)
      ;; Map key type
      (,(concat "\\<map\\s *\\[" type-name) 1 font-lock-type-face)
      ;; Channel value type
      (,(concat "\\<chan\\>\\s *\\(?:<-\\)?" type-name) 1 font-lock-type-face)
      ;; new/make type
      (,(concat "\\<\\(?:new\\|make\\)\\>\\(?:\\s \\|)\\)*(" type-name) 1 font-lock-type-face)
      ;; Type conversion
      (,(concat "\\.\\s *(" type-name) 1 font-lock-type-face)
      ;; Method receiver type
      (,(concat "\\<func\\>\\s *(\\w+\\s +" type-name) 1 font-lock-type-face)
      ;; Labels
      ;; XXX Not quite right.  Also marks compound literal fields.
      ("^\\s *\\(\\w+\\)\\s *:\\(\\S.\\|$\\)" 1 font-lock-constant-face)
      ("\\<\\(goto\\|break\\|continue\\)\\>\\s *\\(\\w+\\)" 2 font-lock-constant-face)))
  "Basic font lock keywords for Go mode.  Highlights keywords,
built-ins, functions, and some types.")

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Key map
;;

(defvar go-mode-map
  (let ((m (make-sparse-keymap)))
    (define-key m "}" #'go-mode-insert-and-indent)
    (define-key m ")" #'go-mode-insert-and-indent)
    (define-key m "," #'go-mode-insert-and-indent)
    (define-key m ":" #'go-mode-delayed-electric)
    ;; In case we get : indentation wrong, correct ourselves
    (define-key m "=" #'go-mode-insert-and-indent)
    m)
  "Keymap used by Go mode to implement electric keys.")

(defun go-mode-insert-and-indent (key)
  "Invoke the global binding of KEY, then reindent the line."

  (interactive (list (this-command-keys)))
  (call-interactively (lookup-key (current-global-map) key))
  (indent-according-to-mode))

(defvar go-mode-delayed-point nil
  "The point following the previous insertion if the insertion
was a delayed electric key.  Used to communicate between
`go-mode-delayed-electric' and `go-mode-delayed-electric-hook'.")
(make-variable-buffer-local 'go-mode-delayed-point)

(defun go-mode-delayed-electric (p)
  "Perform electric insertion, but delayed by one event.

This inserts P into the buffer, as usual, then waits for another key.
If that second key causes a buffer modification starting at the
point after the insertion of P, reindents the line containing P."

  (interactive "p")
  (self-insert-command p)
  (setq go-mode-delayed-point (point)))

(defun go-mode-delayed-electric-hook (b e l)
  "An after-change-function that implements `go-mode-delayed-electric'."

  (when (and go-mode-delayed-point
             (= go-mode-delayed-point b))
    (save-excursion
      (save-match-data
        (goto-char go-mode-delayed-point)
        (indent-according-to-mode))))
  (setq go-mode-delayed-point nil))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Parser
;;

(defvar go-mode-mark-cs-end 1
  "The point at which the comment/string cache ends.  The buffer
will be marked from the beginning up to this point (that is, up
to and including character (1- go-mode-mark-cs-end)).")
(make-variable-buffer-local 'go-mode-mark-cs-end)

(defvar go-mode-mark-string-end 1
  "The point at which the string cache ends.  The buffer
will be marked from the beginning up to this point (that is, up
to and including character (1- go-mode-mark-string-end)).")
(make-variable-buffer-local 'go-mode-mark-string-end)

(defvar go-mode-mark-comment-end 1
  "The point at which the comment cache ends.  The buffer
will be marked from the beginning up to this point (that is, up
to and including character (1- go-mode-mark-comment-end)).")
(make-variable-buffer-local 'go-mode-mark-comment-end)

(defvar go-mode-mark-nesting-end 1
  "The point at which the nesting cache ends.  The buffer will be
marked from the beginning up to this point.")
(make-variable-buffer-local 'go-mode-mark-nesting-end)

(defun go-mode-mark-clear-cs (b e l)
  "An after-change-function that removes the go-mode-cs text property"
  (remove-text-properties b e '(go-mode-cs)))

(defun go-mode-mark-clear-cache (b e)
  "A before-change-function that clears the comment/string and
nesting caches from the modified point on."

  (save-restriction
    (widen)
    (when (<= b go-mode-mark-cs-end)
      ;; Remove the property adjacent to the change position.
      ;; It may contain positions pointing beyond the new end mark.
      (let ((b (let ((cs (get-text-property (max 1 (1- b)) 'go-mode-cs)))
		 (if cs (car cs) b))))
	(remove-text-properties
	 b (min go-mode-mark-cs-end (point-max)) '(go-mode-cs nil))
	(setq go-mode-mark-cs-end b)))

    (when (<= b go-mode-mark-string-end)
      ;; Remove the property adjacent to the change position.
      ;; It may contain positions pointing beyond the new end mark.
      (let ((b (let ((cs (get-text-property (max 1 (1- b)) 'go-mode-string)))
		 (if cs (car cs) b))))
	(remove-text-properties
	 b (min go-mode-mark-string-end (point-max)) '(go-mode-string nil))
	(setq go-mode-mark-string-end b)))
    (when (<= b go-mode-mark-comment-end)
      ;; Remove the property adjacent to the change position.
      ;; It may contain positions pointing beyond the new end mark.
      (let ((b (let ((cs (get-text-property (max 1 (1- b)) 'go-mode-comment)))
		 (if cs (car cs) b))))
	(remove-text-properties
	 b (min go-mode-mark-string-end (point-max)) '(go-mode-comment nil))
	(setq go-mode-mark-comment-end b)))

    (when (< b go-mode-mark-nesting-end)
      (remove-text-properties b (min go-mode-mark-nesting-end (point-max)) '(go-mode-nesting nil))
      (setq go-mode-mark-nesting-end b))))

(defmacro go-mode-parser (&rest body)
  "Evaluate BODY in an environment set up for parsers that use
text properties to mark text.  This inhibits changes to the undo
list or the buffer's modification status and inhibits calls to
the modification hooks.  It also saves the excursion and
restriction and widens the buffer, since most parsers are
context-sensitive."

  (let ((modified-var (make-symbol "modified")))
    `(let ((buffer-undo-list t)
           (,modified-var (buffer-modified-p))
           (inhibit-modification-hooks t)
           (inhibit-read-only t))
       (save-excursion
         (save-restriction
           (widen)
           (unwind-protect
               (progn ,@body)
             (set-buffer-modified-p ,modified-var)))))))

(defun go-mode-cs (&optional pos)
  "Return the comment/string state at point POS.  If point is
inside a comment or string (including the delimiters), this
returns a pair (START . END) indicating the extents of the
comment or string."

  (unless pos
    (setq pos (point)))
  (when (>= pos go-mode-mark-cs-end)
    (go-mode-mark-cs (1+ pos)))
  (get-text-property pos 'go-mode-cs))

(defun go-mode-mark-cs (end)
  "Mark comments and strings up to point END.  Don't call this
directly; use `go-mode-cs'."
  (setq end (min end (point-max)))
  (go-mode-parser
   (save-match-data
     (let ((pos
	    ;; Back up to the last known state.
	    (let ((last-cs
		   (and (> go-mode-mark-cs-end 1)
			(get-text-property (1- go-mode-mark-cs-end)
					   'go-mode-cs))))
	      (if last-cs
		  (car last-cs)
		(max 1 (1- go-mode-mark-cs-end))))))
       (while (< pos end)
	 (goto-char pos)
	 (let ((cs-end			; end of the text property
		(cond
		 ((looking-at "//")
		  (end-of-line)
		  (1+ (point)))
		 ((looking-at "/\\*")
		  (goto-char (+ pos 2))
		  (if (search-forward "*/" (1+ end) t)
		      (point)
		    end))
		 ((looking-at "\"")
		  (goto-char (1+ pos))
		  (if (looking-at "[^\"\n\\\\]*\\(\\\\.[^\"\n\\\\]*\\)*\"")
		      (match-end 0)
		    (end-of-line)
		    (point)))
		 ((looking-at "'")
		  (goto-char (1+ pos))
		  (if (looking-at "[^'\n\\\\]*\\(\\\\.[^'\n\\\\]*\\)*'")
		      (match-end 0)
		    (end-of-line)
		    (point)))
		 ((looking-at "`")
		  (goto-char (1+ pos))
		  (while (if (search-forward "`" end t)
			     (if (eq (char-after) ?`)
				 (goto-char (1+ (point))))
			   (goto-char end)
			   nil))
		  (point)))))
	   (cond
	    (cs-end
	     (put-text-property pos cs-end 'go-mode-cs (cons pos cs-end))
	     (setq pos cs-end))
	    ((re-search-forward "[\"'`]\\|/[/*]" end t)
	     (setq pos (match-beginning 0)))
	    (t
	     (setq pos end)))))
       (setq go-mode-mark-cs-end pos)))))

(defun go-mode-in-comment (&optional pos)
  "Return the comment/string state at point POS.  If point is
inside a comment (including the delimiters), this
returns a pair (START . END) indicating the extents of the
comment or string."

  (unless pos
    (setq pos (point)))
  (when (> pos go-mode-mark-comment-end)
    (go-mode-mark-comment pos))
  (get-text-property pos 'go-mode-comment))

(defun go-mode-mark-comment (end)
  "Mark comments up to point END.  Don't call this directly; use `go-mode-in-comment'."
  (setq end (min end (point-max)))
  (go-mode-parser
   (save-match-data
     (let ((pos
	    ;; Back up to the last known state.
	    (let ((last-comment
		   (and (> go-mode-mark-comment-end 1)
			(get-text-property (1- go-mode-mark-comment-end)
					   'go-mode-comment))))
	      (if last-comment
		  (car last-comment)
		(max 1 (1- go-mode-mark-comment-end))))))
       (while (< pos end)
	 (goto-char pos)
	 (let ((comment-end			; end of the text property
		(cond
		 ((looking-at "//")
		  (end-of-line)
		  (1+ (point)))
		 ((looking-at "/\\*")
		  (goto-char (+ pos 2))
		  (if (search-forward "*/" (1+ end) t)
		      (point)
		    end)))))
	   (cond
	    (comment-end
	     (put-text-property pos comment-end 'go-mode-comment (cons pos comment-end))
	     (setq pos comment-end))
	    ((re-search-forward "/[/*]" end t)
	     (setq pos (match-beginning 0)))
	    (t
	     (setq pos end)))))
       (setq go-mode-mark-comment-end pos)))))

(defun go-mode-in-string (&optional pos)
  "Return the string state at point POS.  If point is
inside a string (including the delimiters), this
returns a pair (START . END) indicating the extents of the
comment or string."

  (unless pos
    (setq pos (point)))
  (when (> pos go-mode-mark-string-end)
    (go-mode-mark-string pos))
  (get-text-property pos 'go-mode-string))

(defun go-mode-mark-string (end)
  "Mark strings up to point END.  Don't call this
directly; use `go-mode-in-string'."
  (setq end (min end (point-max)))
  (go-mode-parser
   (save-match-data
     (let ((pos
	    ;; Back up to the last known state.
	    (let ((last-cs
		   (and (> go-mode-mark-string-end 1)
			(get-text-property (1- go-mode-mark-string-end)
					   'go-mode-string))))
	      (if last-cs
		  (car last-cs)
		(max 1 (1- go-mode-mark-string-end))))))
       (while (< pos end)
	 (goto-char pos)
	 (let ((cs-end			; end of the text property
		(cond
		 ((looking-at "\"")
		  (goto-char (1+ pos))
		  (if (looking-at "[^\"\n\\\\]*\\(\\\\.[^\"\n\\\\]*\\)*\"")
		      (match-end 0)
		    (end-of-line)
		    (point)))
		 ((looking-at "'")
		  (goto-char (1+ pos))
		  (if (looking-at "[^'\n\\\\]*\\(\\\\.[^'\n\\\\]*\\)*'")
		      (match-end 0)
		    (end-of-line)
		    (point)))
		 ((looking-at "`")
		  (goto-char (1+ pos))
		  (while (if (search-forward "`" end t)
			     (if (eq (char-after) ?`)
				 (goto-char (1+ (point))))
			   (goto-char end)
			   nil))
		  (point)))))
	   (cond
	    (cs-end
	     (put-text-property pos cs-end 'go-mode-string (cons pos cs-end))
	     (setq pos cs-end))
	    ((re-search-forward "[\"'`]" end t)
	     (setq pos (match-beginning 0)))
	    (t
	     (setq pos end)))))
       (setq go-mode-mark-string-end pos)))))

(defun go-mode-font-lock-cs (limit comment)
  "Helper function for highlighting comment/strings.  If COMMENT is t,
set match data to the next comment after point, and advance point
after it.  If COMMENT is nil, use the next string.  Returns nil
if no further tokens of the type exist."
  ;; Ensures that `next-single-property-change' below will work properly.
  (go-mode-cs limit)
  (let (cs next (result 'scan))
    (while (eq result 'scan)
      (if (or (>= (point) limit) (eobp))
	  (setq result nil)
	(setq cs (go-mode-cs))
	(if cs
	    (if (eq (= (char-after (car cs)) ?/) comment)
		;; If inside the expected comment/string, highlight it.
		(progn
		  ;; If the match includes a "\n", we have a
		  ;; multi-line construct.  Mark it as such.
		  (goto-char (car cs))
		  (when (search-forward "\n" (cdr cs) t)
		    (put-text-property
		     (car cs) (cdr cs) 'font-lock-multline t))
		  (set-match-data (list (car cs) (copy-marker (cdr cs))))
		  (goto-char (cdr cs))
		  (setq result t))
	      ;; Wrong type.  Look for next comment/string after this one.
	      (goto-char (cdr cs)))
	  ;; Not inside comment/string.  Search for next comment/string.
	  (setq next (next-single-property-change
		      (point) 'go-mode-cs nil limit))
	  (if (and next (< next limit))
	      (goto-char next)
	    (setq result nil)))))
    result))

(defun go-mode-font-lock-cs-string (limit)
  "Font-lock iterator for strings."
  (go-mode-font-lock-cs limit nil))

(defun go-mode-font-lock-cs-comment (limit)
  "Font-lock iterator for comments."
  (go-mode-font-lock-cs limit t))

(defsubst go-mode-nesting (&optional pos)
  "Return the nesting at point POS.  The nesting is a list
of (START . END) pairs for all braces, parens, and brackets
surrounding POS, starting at the inner-most nesting.  START is
the location of the open character.  END is the location of the
close character or nil if the nesting scanner has not yet
encountered the close character."

  (unless pos
    (setq pos (point)))
  (if (= pos 1)
      '()
    (when (> pos go-mode-mark-nesting-end)
      (go-mode-mark-nesting pos))
    (get-text-property (- pos 1) 'go-mode-nesting)))

(defun go-mode-mark-nesting (pos)
  "Mark nesting up to point END.  Don't call this directly; use
`go-mode-nesting'."

  (go-mode-cs pos)
  (go-mode-parser
   ;; Mark depth
   (goto-char go-mode-mark-nesting-end)
   (let ((nesting (go-mode-nesting))
         (last (point)))
     (while (< last pos)
       ;; Find the next depth-changing character
       (skip-chars-forward "^(){}[]" pos)
       ;; Mark everything up to this character with the current
       ;; nesting
       (put-text-property last (point) 'go-mode-nesting nesting)
       (when nil
         (let ((depth (length nesting)))
           (put-text-property last (point) 'face
                              `((:background
                                 ,(format "gray%d" (* depth 10)))))))
       (setq last (point))
       ;; Update nesting
       (unless (eobp)
         (let ((ch (unless (go-mode-cs) (char-after))))
           (forward-char 1)
           (case ch
             ((?\( ?\{ ?\[)
              (setq nesting (cons (cons (- (point) 1) nil)
                                  nesting)))
             ((?\) ?\} ?\])
              (when nesting
                (setcdr (car nesting) (- (point) 1))
                (setq nesting (cdr nesting))))))))
     ;; Update state
     (setq go-mode-mark-nesting-end last))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Indentation
;;

(defvar go-mode-non-terminating-keywords-regexp
  (let* ((kws go-mode-keywords)
         (kws (remove "break" kws))
         (kws (remove "continue" kws))
         (kws (remove "fallthrough" kws))
         (kws (remove "return" kws)))
    (regexp-opt kws 'words))
  "Regular expression matching all Go keywords that *do not*
implicitly terminate a statement.")

(defun go-mode-semicolon-p ()
  "True iff point immediately follows either an explicit or
implicit semicolon.  Point should immediately follow the last
token on the line."

  ;; #Semicolons
  (case (char-before)
    ((?\;) t)
    ;; String literal
    ((?' ?\" ?`) t)
    ;; One of the operators and delimiters ++, --, ), ], or }
    ((?+) (eq (char-before (1- (point))) ?+))
    ((?-) (eq (char-before (1- (point))) ?-))
    ((?\) ?\] ?\}) t)
    ;; An identifier or one of the keywords break, continue,
    ;; fallthrough, or return or a numeric literal
    (otherwise
     (save-excursion
       (when (/= (skip-chars-backward "[:word:]_") 0)
         (not (looking-at go-mode-non-terminating-keywords-regexp)))))))

(defun go-mode-whitespace-p (char)
  "Is newline, or char whitespace in the syntax table for go."
  (or (eq char ?\n)
      (= (char-syntax char) ?\ )))

(defun go-mode-backward-skip-comments ()
  "Skip backward over comments and whitespace."
  ;; only proceed if point is in a comment or white space
  (if (or (go-mode-in-comment)
	  (go-mode-whitespace-p (char-after (point))))
      (let ((loop-guard t))
	(while (and
		loop-guard
		(not (bobp)))

	  (cond ((go-mode-whitespace-p (char-after (point)))
		 ;; moves point back over any whitespace
		 (re-search-backward "[^[:space:]]"))

		((go-mode-in-comment)
		 ;; move point to char preceeding current comment
		 (goto-char (1- (car (go-mode-in-comment)))))
		
		;; not in a comment or whitespace? we must be done.
		(t (setq loop-guard nil)
		   (forward-char 1)))))))

(defun go-mode-indentation ()
  "Compute the ideal indentation level of the current line.

To the first order, this is the brace depth of the current line,
plus parens that follow certain keywords.  case, default, and
labels are outdented one level, and continuation lines are
indented one level."

  (save-excursion
    (back-to-indentation)
    (let ((cs (go-mode-cs)))
      ;; Treat comments and strings differently only if the beginning
      ;; of the line is contained within them
      (when (and cs (= (point) (car cs)))
        (setq cs nil))
      ;; What type of context am I in?
      (cond
       ((and cs (save-excursion
                  (goto-char (car cs))
                  (looking-at "`")))
        ;; Inside a multi-line string.  Don't mess with indentation.
        nil)
       (cs
        ;; Inside a general comment
        (goto-char (car cs))
        (forward-char 1)
        (current-column))
       (t
        ;; Not in a multi-line string or comment
        (let ((indent 0)
              (inside-indenting-paren nil))
          ;; Count every enclosing brace, plus parens that follow
          ;; import, const, var, or type and indent according to
          ;; depth.  This simple rule does quite well, but also has a
          ;; very large extent.  It would be better if we could mimic
          ;; some nearby indentation.
          (save-excursion
            (skip-chars-forward "})")
            (let ((first t))
              (dolist (nest (go-mode-nesting))
                (case (char-after (car nest))
                  ((?\{)
                   (incf indent tab-width))
                  ((?\()
                   (goto-char (car nest))
                   (go-mode-backward-skip-comments)
                   (backward-char)
                   ;; Really just want the token before
                   (when (looking-back "\\<import\\|const\\|var\\|type\\|package"
                                       (max (- (point) 7) (point-min)))
                     (incf indent tab-width)
                     (when first
                       (setq inside-indenting-paren t)))))
                (setq first nil))))

          ;; case, default, and labels are outdented 1 level
          (when (looking-at "\\<case\\>\\|\\<default\\>\\|\\w+\\s *:\\(\\S.\\|$\\)")
            (decf indent tab-width))

	  (when (looking-at "\\w+\\s *:.+,\\s *$")
	    (incf indent tab-width))

          ;; Continuation lines are indented 1 level
          (beginning-of-line)		; back up to end of previous line
	  (backward-char)
          (go-mode-backward-skip-comments) ; back up past any comments
          (when (case (char-before)
                  ((nil ?\{ ?:)
                   ;; At the beginning of a block or the statement
                   ;; following a label.
                   nil)
                  ((?\()
                   ;; Usually a continuation line in an expression,
                   ;; unless this paren is part of a factored
                   ;; declaration.
                   (not inside-indenting-paren))
                  ((?,)
                   ;; Could be inside a literal.  We're a little
                   ;; conservative here and consider any comma within
                   ;; curly braces (as opposed to parens) to be a
                   ;; literal separator.  This will fail to recognize
                   ;; line-breaks in parallel assignments as
                   ;; continuation lines.
                   (let ((depth (go-mode-nesting)))
                     (and depth
                          (not (eq (char-after (caar depth)) ?\{)))))
                  (t
                   ;; We're in the middle of a block.  Did the
                   ;; previous line end with an implicit or explicit
                   ;; semicolon?
                   (not (go-mode-semicolon-p))))
            (incf indent tab-width))

          (max indent 0)))))))

(defun go-mode-indent-line ()
  "Indent the current line according to `go-mode-indentation'."
  (interactive)

  ;; turn off case folding to distinguish keywords from identifiers
  ;; e.g. "default" is a keyword; "Default" can be a variable name.
  (let ((case-fold-search nil))
    (let ((col (go-mode-indentation)))
      (when col
	(let ((offset (- (current-column) (current-indentation))))
	  (indent-line-to col)
	  (when (> offset 0)
	    (forward-char offset)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Go mode
;;

;;;###autoload
(define-derived-mode go-mode nil "Go"
  "Major mode for editing Go source text.

This provides basic syntax highlighting for keywords, built-ins,
functions, and some types.  It also provides indentation that is
\(almost) identical to gofmt."

  ;; Font lock
  (set (make-local-variable 'font-lock-defaults)
       '(go-mode-font-lock-keywords nil nil nil nil))

  ;; Remove stale text properties
  (save-restriction
    (widen)
    (let ((modified (buffer-modified-p)))
      (remove-text-properties 1 (point-max)
                              '(go-mode-cs nil go-mode-nesting nil))
      ;; remove-text-properties marks the buffer modified. undo that if it
      ;; wasn't originally marked modified.
      (set-buffer-modified-p modified)))

  ;; Reset the syntax mark caches
  (setq go-mode-mark-cs-end      1
        go-mode-mark-nesting-end 1)
  (add-hook 'before-change-functions #'go-mode-mark-clear-cache nil t)
  (add-hook 'after-change-functions #'go-mode-mark-clear-cs nil t)

  ;; Indentation
  (set (make-local-variable 'indent-line-function)
       #'go-mode-indent-line)
  (add-hook 'after-change-functions #'go-mode-delayed-electric-hook nil t)

  ;; Comments
  (set (make-local-variable 'comment-start) "// ")
  (set (make-local-variable 'comment-end)   "")

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
(add-to-list 'auto-mode-alist (cons "\\.go$" #'go-mode))

(defun go-mode-reload ()
  "Reload go-mode.el and put the current buffer into Go mode.
Useful for development work."

  (interactive)
  (unload-feature 'go-mode)
  (require 'go-mode)
  (go-mode))

;;;###autoload
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
              (coding-system-for-read 'utf-8)    ;; use utf-8 with subprocesses
              (coding-system-for-write 'utf-8))
          (with-current-buffer errbuf
            (toggle-read-only 0)
            (erase-buffer))
          (with-current-buffer srcbuf
            (save-restriction
              (let (deactivate-mark)
                (widen)
                ; If this is a new file, diff-mode can't apply a
                ; patch to a non-exisiting file, so replace the buffer
                ; completely with the output of 'gofmt'.
                ; If the file exists, patch it to keep the 'undo' list happy.
                (let* ((newfile (not (file-exists-p filename)))
                      (flag (if newfile "" " -d")))
                  (if (= 0 (shell-command-on-region (point-min) (point-max)
                                                    (concat "gofmt" flag)
                                                    patchbuf nil errbuf))
                      ; gofmt succeeded: replace buffer or apply patch hunks.
                      (let ((old-point (point))
                            (old-mark (mark t)))
                        (kill-buffer errbuf)
                        (if newfile
                            ; New file, replace it (diff-mode won't work)
                            (gofmt-replace-buffer srcbuf patchbuf)
                          ; Existing file, patch it
                          (gofmt-apply-patch filename srcbuf patchbuf))
                        (goto-char (min old-point (point-max)))
                        ;; Restore the mark and point
                        (if old-mark (push-mark (min old-mark (point-max)) t))
                        (set-window-configuration currconf))

                  ;; gofmt failed: display the errors
                  (gofmt-process-errors filename errbuf))))))

          ;; Collapse any window opened on outbuf if shell-command-on-region
          ;; displayed it.
          (delete-windows-on patchbuf)))
      (kill-buffer patchbuf))))

(defun gofmt-replace-buffer (srcbuf patchbuf)
  (with-current-buffer srcbuf
    (erase-buffer)
    (insert-buffer-substring patchbuf)))

(defconst gofmt-stdin-tag "<standard input>")

(defun gofmt-apply-patch (filename srcbuf patchbuf)
  (require 'diff-mode)
  ;; apply all the patch hunks
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
          (diff-apply-hunk))
      ;; When there's no more hunks, diff-hunk-next signals an error, ignore it
      (error nil))))

(defun gofmt-process-errors (filename errbuf)
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
 (add-hook 'before-save-hook #'gofmt-before-save)"

  (interactive)
  (when (eq major-mode 'go-mode) (gofmt)))

(defun godoc-read-query ()
  "Read a godoc query from the minibuffer."
  ;; Compute the default query as the symbol under the cursor.
  ;; TODO: This does the wrong thing for e.g. multipart.NewReader (it only grabs
  ;; half) but I see no way to disambiguate that from e.g. foobar.SomeMethod.
  (let* ((bounds (bounds-of-thing-at-point 'symbol))
         (symbol (if bounds
                     (buffer-substring-no-properties (car bounds)
                                                     (cdr bounds)))))
    (read-string (if symbol
                     (format "godoc (default %s): " symbol)
                   "godoc: ")
                 nil nil symbol)))

(defun godoc-get-buffer (query)
  "Get an empty buffer for a godoc query."
  (let* ((buffer-name (concat "*godoc " query "*"))
         (buffer (get-buffer buffer-name)))
    ;; Kill the existing buffer if it already exists.
    (when buffer (kill-buffer buffer))
    (get-buffer-create buffer-name)))

(defun godoc-buffer-sentinel (proc event)
  "Sentinel function run when godoc command completes."
  (with-current-buffer (process-buffer proc)
    (cond ((string= event "finished\n")  ;; Successful exit.
           (goto-char (point-min))
           (display-buffer (current-buffer) 'not-this-window))
          ((not (= (process-exit-status proc) 0))  ;; Error exit.
           (let ((output (buffer-string)))
             (kill-buffer (current-buffer))
             (message (concat "godoc: " output)))))))

;;;###autoload
(defun godoc (query)
  "Show go documentation for a query, much like M-x man."
  (interactive (list (godoc-read-query)))
  (unless (string= query "")
    (set-process-sentinel
     (start-process-shell-command "godoc" (godoc-get-buffer query)
                                  (concat "godoc " query))
     'godoc-buffer-sentinel)
    nil))

(provide 'go-mode)
