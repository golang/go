# Vim / Neovim

## vim-go

Use [vim-go] ver 1.20+, with the following configuration:

```
let g:go_def_mode='gopls'
let g:go_info_mode='gopls'
```

## LanguageClient-neovim

Use [LanguageClient-neovim], with the following configuration:

```
" Launch gopls when Go files are in use
let g:LanguageClient_serverCommands = {
       \ 'go': ['gopls']
       \ }
" Run gofmt on save
autocmd BufWritePre *.go :call LanguageClient#textDocument_formatting_sync()
```

## Ale

Use [ale]:

```vim
let g:ale_linters = {
	\ 'go': ['gopls'],
	\}
```

see [this issue][ale-issue-2179]

## vim-lsp

Use [prabirshrestha/vim-lsp], with the following configuration:

```vim
augroup LspGo
  au!
  autocmd User lsp_setup call lsp#register_server({
      \ 'name': 'go-lang',
      \ 'cmd': {server_info->['gopls']},
      \ 'whitelist': ['go'],
      \ })
  autocmd FileType go setlocal omnifunc=lsp#complete
  "autocmd FileType go nmap <buffer> gd <plug>(lsp-definition)
  "autocmd FileType go nmap <buffer> ,n <plug>(lsp-next-error)
  "autocmd FileType go nmap <buffer> ,p <plug>(lsp-previous-error)
augroup END
```

## vim-lsc

Use [natebosch/vim-lsc], with the following configuration:

```vim
let g:lsc_server_commands = {
\  "go": {
\    "command": "gopls serve",
\    "log_level": -1,
\    "suppress_stderr": v:true,
\  },
\}
```

The `log_level` and `suppress_stderr` parts are needed to prevent breakage from logging. See
issues [#180](https://github.com/natebosch/vim-lsc/issues/180) and
[#213](https://github.com/natebosch/vim-lsc/issues/213).

## coc.nvim

Use [coc.nvim], with the following `coc-settings.json` configuration:

```json
  "languageserver": {
    "golang": {
      "command": "gopls",
      "rootPatterns": ["go.mod", ".vim/", ".git/", ".hg/"],
      "filetypes": ["go"],
      "initializationOptions": {
        "usePlaceholders": true
      }
    }
  }
```

Other [settings](settings.md) can be added in `initializationOptions` too.

The `editor.action.organizeImport` code action will auto-format code and add missing imports. To run this automatically on save, add the following line to your `init.vim`:

```vim
autocmd BufWritePre *.go :call CocAction('runCommand', 'editor.action.organizeImport')
```

## govim

In vim classic only, use the experimental [`govim`], simply follow the [install steps][govim-install].

[vim-go]: https://github.com/fatih/vim-go
[LanguageClient-neovim]: https://github.com/autozimu/LanguageClient-neovim
[ale]: https://github.com/w0rp/ale
[ale-issue-2179]: https://github.com/w0rp/ale/issues/2179
[prabirshrestha/vim-lsp]: https://github.com/prabirshrestha/vim-lsp/
[natebosch/vim-lsc]: https://github.com/natebosch/vim-lsc/
[natebosch/vim-lsc#180]: https://github.com/natebosch/vim-lsc/issues/180
[coc.nvim]: https://github.com/neoclide/coc.nvim/
[`govim`]: https://github.com/myitcv/govim
[govim-install]: https://github.com/myitcv/govim/blob/master/README.md#govim---go-development-plugin-for-vim8
