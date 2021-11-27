# Vim / Neovim

* [vim-go](#vimgo)
* [LanguageClient-neovim](#lcneovim)
* [Ale](#ale)
* [vim-lsp](#vimlsp)
* [vim-lsc](#vimlsc)
* [coc.nvim](#cocnvim)
* [govim](#govim)
* [Neovim v0.5.0+](#neovim)
  * [Installation](#neovim-install)
  * [Custom Configuration](#neovim-config)
  * [Imports](#neovim-imports)
  * [Omnifunc](#neovim-omnifunc)
  * [Additional Links](#neovim-links)

## <a href="#vimgo" id="vimgo">vim-go</a>

Use [vim-go] ver 1.20+, with the following configuration:

```vim
let g:go_def_mode='gopls'
let g:go_info_mode='gopls'
```

## <a href="#lcneovim" id="lcneovim">LanguageClient-neovim</a>

Use [LanguageClient-neovim], with the following configuration:

```vim
" Launch gopls when Go files are in use
let g:LanguageClient_serverCommands = {
       \ 'go': ['gopls']
       \ }
" Run gofmt on save
autocmd BufWritePre *.go :call LanguageClient#textDocument_formatting_sync()
```

## <a href="#ale" id="ale">Ale</a>

Use [ale]:

```vim
let g:ale_linters = {
  \ 'go': ['gopls'],
  \}
```

see [this issue][ale-issue-2179]

## <a href="#vimlsp" id="vimlsp">vim-lsp</a>

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

## <a href="#vimlsc" id="vimlsc">vim-lsc</a>

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

## <a href="#cocnvim" id="cocnvim">coc.nvim</a>

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

## <a href="#govim" id="govim">govim</a>

In vim classic only, use the experimental [`govim`], simply follow the [install steps][govim-install].

## <a href="#neovim" id="neovim">Neovim v0.5.0+</a>

To use the new native LSP client in Neovim, make sure you
[install][nvim-install] Neovim v.0.5.0+,
the `nvim-lspconfig` configuration helper plugin, and check the
[`gopls` configuration section][nvim-lspconfig] there.

### <a href="#neovim-install" id="neovim-install">Installation</a>

You can use Neovim's native plugin system.  On a Unix system, you can do that by
cloning the `nvim-lspconfig` repository into the correct directory:

```sh
dir="${HOME}/.local/share/nvim/site/pack/nvim-lspconfig/opt/nvim-lspconfig/"
mkdir -p "$dir"
cd "$dir"
git clone 'https://github.com/neovim/nvim-lspconfig.git' .
```

### <a href="#neovim-config" id="neovim-config">Custom Configuration</a>

You can add custom configuration using Lua.  Here is an example of enabling the
`unusedparams` check as well as `staticcheck`:

```vim
lua <<EOF
  lspconfig = require "lspconfig"
  lspconfig.gopls.setup {
    cmd = {"gopls", "serve"},
    settings = {
      gopls = {
        analyses = {
          unusedparams = true,
        },
        staticcheck = true,
      },
    },
  }
EOF
```

### <a href="#neovim-imports" id="neovim-imports">Imports</a>

To get your imports ordered on save, like `goimports` does, you can define
a helper function in Lua:

```vim
lua <<EOF
  -- …

  function goimports(timeout_ms)
    local context = { only = { "source.organizeImports" } }
    vim.validate { context = { context, "t", true } }

    local params = vim.lsp.util.make_range_params()
    params.context = context

    -- See the implementation of the textDocument/codeAction callback
    -- (lua/vim/lsp/handler.lua) for how to do this properly.
    local result = vim.lsp.buf_request_sync(0, "textDocument/codeAction", params, timeout_ms)
    if not result or next(result) == nil then return end
    local actions = result[1].result
    if not actions then return end
    local action = actions[1]

    -- textDocument/codeAction can return either Command[] or CodeAction[]. If it
    -- is a CodeAction, it can have either an edit, a command or both. Edits
    -- should be executed first.
    if action.edit or type(action.command) == "table" then
      if action.edit then
        vim.lsp.util.apply_workspace_edit(action.edit)
      end
      if type(action.command) == "table" then
        vim.lsp.buf.execute_command(action.command)
      end
    else
      vim.lsp.buf.execute_command(action)
    end
  end
EOF

autocmd BufWritePre *.go lua goimports(1000)
```

(Taken from the [discussion][nvim-lspconfig-imports] on Neovim issue tracker.)

### <a href="#neovim-omnifunc" id="neovim-omnifunc">Omnifunc</a>

To make your <kbd>Ctrl</kbd>+<kbd>x</kbd>,<kbd>Ctrl</kbd>+<kbd>o</kbd> work, add
this to your `init.vim`:

```vim
autocmd FileType go setlocal omnifunc=v:lua.vim.lsp.omnifunc
```

### <a href="#neovim-links" id="neovim-links">Additional Links</a>

* [Neovim's official LSP documentation][nvim-docs].

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
[nvim-docs]: https://neovim.io/doc/user/lsp.html
[nvim-install]: https://github.com/neovim/neovim/wiki/Installing-Neovim
[nvim-lspconfig]: https://github.com/neovim/nvim-lspconfig/blob/master/doc/server_configurations.md#gopls
[nvim-lspconfig-imports]: https://github.com/neovim/nvim-lspconfig/issues/115
