" Copyright 2011 The Go Authors. All rights reserved.
" Use of this source code is governed by a BSD-style
" license that can be found in the LICENSE file.
"
" import.vim: Vim commands to import/drop Go packages.
"
" This filetype plugin adds three new commands for go buffers:
"
"   :Import {path}
"
"       Import ensures that the provided package {path} is imported
"       in the current Go buffer, using proper style and ordering.
"       If {path} is already being imported, an error will be
"       displayed and the buffer will be untouched.
" 
"   :ImportAs {localname} {path}
"
"       Same as Import, but uses a custom local name for the package.
"
"   :Drop {path}
"
"       Remove the import line for the provided package {path}, if
"       present in the current Go buffer.  If {path} is not being
"       imported, an error will be displayed and the buffer will be
"       untouched.
"
" In addition to these commands, there are also two shortcuts mapped:
"
"   \f  -  Runs :Import fmt
"   \F  -  Runs :Drop fmt
"
" The backslash is the default maplocalleader, so it is possible that
" your vim is set to use a different character (:help maplocalleader).
"
if exists("b:did_ftplugin")
    finish
endif

command! -buffer -nargs=? Drop call s:SwitchImport(0, '', <f-args>)
command! -buffer -nargs=1 Import call s:SwitchImport(1, '', <f-args>)
command! -buffer -nargs=* ImportAs call s:SwitchImport(1, <f-args>)
map <buffer> <LocalLeader>f :Import fmt<CR>
map <buffer> <LocalLeader>F :Drop fmt<CR>

function! s:SwitchImport(enabled, localname, path)
    let view = winsaveview()
    let path = a:path

    " Quotes are not necessary, so remove them if provided.
    if path[0] == '"'
        let path = strpart(path, 1)
    endif
    if path[len(path)-1] == '"'
        let path = strpart(path, 0, len(path) - 1)
    endif
    if path == ''
        call s:Error('Import path not provided')
        return
    endif

    let qpath = '"' . path . '"'
    if a:localname != ''
        let qlocalpath = a:localname . ' ' . qpath
    else
        let qlocalpath = qpath
    endif
    let indentstr = 0
    let packageline = -1 " Position of package name statement
    let appendline = -1  " Position to introduce new import
    let deleteline = -1  " Position of line with existing import
    let linesdelta = 0   " Lines added/removed

    " Find proper place to add/remove import.
    let line = 0
    while line <= line('$')
        let linestr = getline(line)

        if linestr =~# '^package\s'
            let packageline = line
            let appendline = line

        elseif linestr =~# '^import\s\+('
            let appendstr = qlocalpath
            let indentstr = 1
            let appendline = line
            while line <= line("$")
                let line = line + 1
                let linestr = getline(line)
                let m = matchlist(getline(line), '^\()\|\(\s\+\)\(\S*\s*\)"\(.\+\)"\)')
                if empty(m)
                    continue
                endif
                if m[1] == ')'
                    break
                endif
                if a:localname != '' && m[3] != ''
                    let qlocalpath = printf('%-' . (len(m[3])-1) . 's %s', a:localname, qpath)
                endif
                let appendstr = m[2] . qlocalpath
                let indentstr = 0
                if m[4] == path
                    let appendline = -1
                    let deleteline = line
                    break
                elseif m[4] < path
                    let appendline = line
                endif
            endwhile
            break

        elseif linestr =~# '^import '
            if appendline == packageline
                let appendstr = 'import ' . qlocalpath
                let appendline = line - 1
            endif
            let m = matchlist(linestr, '^import\(\s\+\)\(\S*\s*\)"\(.\+\)"')
            if !empty(m)
                if m[3] == path
                    let appendline = -1
                    let deleteline = line
                    break
                endif
                if m[3] < path
                    let appendline = line
                endif
                if a:localname != '' && m[2] != ''
                    let qlocalpath = printf("%s %" . len(m[2])-1 . "s", a:localname, qpath)
                endif
                let appendstr = 'import' . m[1] . qlocalpath
            endif

        elseif linestr =~# '^\(var\|const\|type\|func\)\>'
            break

        endif
        let line = line + 1
    endwhile

    " Append or remove the package import, as requested.
    if a:enabled
        if deleteline != -1
            call s:Error(qpath . ' already being imported')
        elseif appendline == -1
            call s:Error('No package line found')
        else
            if appendline == packageline
                call append(appendline + 0, '')
                call append(appendline + 1, 'import (')
                call append(appendline + 2, ')')
                let appendline += 2
                let linesdelta += 3
                let appendstr = qlocalpath
                let indentstr = 1
            endif
            call append(appendline, appendstr)
            execute appendline + 1
            if indentstr
                execute 'normal >>'
            endif
            let linesdelta += 1
        endif
    else
        if deleteline == -1
            call s:Error(qpath . ' not being imported')
        else
            execute deleteline . 'd'
            let linesdelta -= 1

            if getline(deleteline-1) =~# '^import\s\+(' && getline(deleteline) =~# '^)'
                " Delete empty import block
                let deleteline -= 1
                execute deleteline . "d"
                execute deleteline . "d"
                let linesdelta -= 2
            endif

            if getline(deleteline) == '' && getline(deleteline - 1) == ''
                " Delete spacing for removed line too.
                execute deleteline . "d"
                let linesdelta -= 1
            endif
        endif
    endif

    " Adjust view for any changes.
    let view.lnum += linesdelta
    let view.topline += linesdelta
    if view.topline < 0
        let view.topline = 0
    endif

    " Put buffer back where it was.
    call winrestview(view)

endfunction

function! s:Error(s)
    echohl Error | echo a:s | echohl None
endfunction

" vim:ts=4:sw=4:et
