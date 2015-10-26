;;; go-rename.el --- Integration of the 'gorename' tool into Emacs.

;; Copyright 2014 The Go Authors. All rights reserved.
;; Use of this source code is governed by a BSD-style
;; license that can be found in the LICENSE file.

;; Version: 0.1
;; Package-Requires: ((go-mode "1.3.1"))
;; Keywords: tools

;;; Commentary:

;; To install:

;; % go get golang.org/x/tools/cmd/gorename
;; % go build golang.org/x/tools/cmd/gorename
;; % mv gorename $HOME/bin/         # or elsewhere on $PATH

;; The go-rename-command variable can be customized to specify an
;; alternative location for the installed command.

;;; Code:

(require 'compile)
(require 'go-mode)
(require 'thingatpt)

(defgroup go-rename nil
  "Options specific to the Go rename."
  :group 'go)

(defcustom go-rename-command "gorename"
  "The `gorename' command; by the default, $PATH is searched."
  :type 'string
  :group 'go-rename)

;;;###autoload
(defun go-rename (new-name &optional force)
  "Rename the entity denoted by the identifier at point, using
the `gorename' tool. With FORCE, call `gorename' with the
`-force' flag."
  (interactive (list (read-string "New name: " (thing-at-point 'symbol))
                     current-prefix-arg))
  (if (not buffer-file-name)
      (error "Cannot use go-rename on a buffer without a file name"))
  ;; It's not sufficient to save the current buffer if modified,
  ;; since if gofmt-before-save is on the before-save-hook,
  ;; saving will disturb the selected region.
  (if (buffer-modified-p)
      (error "Please save the current buffer before invoking go-rename"))
  ;; Prompt-save all other modified Go buffers, since they might get written.
  (save-some-buffers nil #'(lambda ()
              (and (buffer-file-name)
                   (string= (file-name-extension (buffer-file-name)) ".go"))))
  (let* ((posflag (format "-offset=%s:#%d"
                          buffer-file-name
                          (1- (go--position-bytes (point)))))
         (env-vars (go-root-and-paths))
         (goroot-env (concat "GOROOT=" (car env-vars)))
         (gopath-env (concat "GOPATH=" (mapconcat #'identity (cdr env-vars) ":")))
         success)
    (with-current-buffer (get-buffer-create "*go-rename*")
      (setq buffer-read-only nil)
      (erase-buffer)
      (let ((args (append (list go-rename-command nil t nil posflag "-to" new-name) (if force '("-force")))))
        ;; Log the command to *Messages*, for debugging.
        (message "Command: %s:" args)
        (message "Running gorename...")
        ;; Use dynamic binding to modify/restore the environment
        (setq success (zerop (let ((process-environment (list* goroot-env gopath-env process-environment)))
          (apply #'call-process args))))
      (insert "\n")
      (compilation-mode)
      (setq compilation-error-screen-columns nil)

      ;; On success, print the one-line result in the message bar,
      ;; and hide the *go-rename* buffer.
      (if success
          (progn
            (message "%s" (go--buffer-string-no-trailing-space))
            (gofmt--kill-error-buffer (current-buffer)))
        ;; failure
        (let ((w (display-buffer (current-buffer))))
          (message "gorename exited")
          (shrink-window-if-larger-than-buffer w)
          (set-window-point w (point-min)))))))

  ;; Reload the modified files, saving line/col.
  ;; (Don't restore the point since the text has changed.)
  ;;
  ;; TODO(adonovan): should we also do this for all other files
  ;; that were updated (the tool can print them)?
  (let ((line (line-number-at-pos))
        (col (current-column)))
    (revert-buffer t t t) ; safe, because we just saved it
    (goto-char (point-min))
    (forward-line (1- line))
    (forward-char col)))


(defun go--buffer-string-no-trailing-space ()
  (replace-regexp-in-string "[\t\n ]*\\'"
                            ""
                            (buffer-substring (point-min) (point-max))))

(provide 'go-rename)

;;; go-rename.el ends here
