;;; go-mode-load.el --- Major mode for the Go programming language

;;; Commentary:

;; To install go-mode, add the following lines to your .emacs file:
;;   (add-to-list 'load-path "PATH CONTAINING go-mode-load.el" t)
;;   (require 'go-mode-load)
;; After this, go-mode will be used for files ending in '.go'.

;; To compile go-mode from the command line, run the following
;;   emacs -batch -f batch-byte-compile go-mode.el

;; See go-mode.el for documentation.

;;; Code:

;; To update this file, evaluate the following form
;;   (let ((generated-autoload-file buffer-file-name)) (update-file-autoloads "go-mode.el"))


;;;### (autoloads (go-mode) "go-mode" "go-mode.el" (19168 32439))
;;; Generated autoloads from go-mode.el

(autoload (quote go-mode) "go-mode" "\
Major mode for editing Go source text.

This provides basic syntax highlighting for keywords, built-ins,
functions, and some types.  It also provides indentation that is
\(almost) identical to gofmt.

\(fn)" t nil)

(add-to-list (quote auto-mode-alist) (cons "\\.go$" (function go-mode)))

;;;***

(provide 'go-mode-load)
