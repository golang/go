// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

'use strict';

import fs = require('fs');
import lsp = require('vscode-languageclient');
import vscode = require('vscode');
import path = require('path');

export function activate(ctx: vscode.ExtensionContext): void {
  let document = vscode.window.activeTextEditor.document;
  let config = vscode.workspace.getConfiguration('golsp', document.uri);
  let golspFlags: string[] = config['flags'];
  let serverOptions:
      lsp.ServerOptions = {command: getBinPath('golsp'), args: golspFlags};
  let clientOptions: lsp.LanguageClientOptions = {
    initializationOptions: {},
    documentSelector: ['go'],
    uriConverters: {
      code2Protocol: (uri: vscode.Uri): string =>
          (uri.scheme ? uri : uri.with({scheme: 'file'})).toString(),
      protocol2Code: (uri: string) => vscode.Uri.parse(uri),
    },
    revealOutputChannelOn: lsp.RevealOutputChannelOn.Never,
  };
  const c = new lsp.LanguageClient('golsp', serverOptions, clientOptions);
  c.onReady().then(() => {
    const capabilities = c.initializeResult && c.initializeResult.capabilities;
    if (!capabilities) {
      return vscode.window.showErrorMessage(
          'The language server is not able to serve any features at the moment.');
    }
  });
  ctx.subscriptions.push(c.start());
}

function getBinPath(toolName: string): string {
  toolName = correctBinname(toolName);
  let tool = findToolIn(toolName, 'PATH', false);
  if (tool) {
    return tool;
  }
  return findToolIn(toolName, 'GOPATH', true);
}

function findToolIn(
    toolName: string, envVar: string, appendBinToPath: boolean): string {
  let value = process.env[envVar];
  if (value) {
    let paths = value.split(path.delimiter);
    for (let i = 0; i < paths.length; i++) {
      let binpath = path.join(paths[i], appendBinToPath ? 'bin' : '', toolName);
      if (fileExists(binpath)) {
        return binpath;
      }
    }
  }
  return null;
}

function fileExists(filePath: string): boolean {
  try {
    return fs.statSync(filePath).isFile();
  } catch (e) {
    return false;
  }
}

function correctBinname(toolName: string) {
  if (process.platform === 'win32')
    return toolName + '.exe';
  else
    return toolName;
}