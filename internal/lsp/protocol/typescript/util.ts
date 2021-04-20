
// for us typescript ignorati, having an import makes this file a module
import * as fs from 'fs';
import * as process from 'process';
import * as ts from 'typescript';

// This file contains various utilities having to do with producing strings
// and managing output

// ------ create files
let dir = process.env['HOME'];
const srcDir = '/vscode-languageserver-node';
export const fnames = [
  `${dir}${srcDir}/protocol/src/common/protocol.ts`,
  `${dir}/${srcDir}/protocol/src/browser/main.ts`, `${dir}${srcDir}/types/src/main.ts`,
  `${dir}${srcDir}/jsonrpc/src/node/main.ts`
];
export const gitHash = 'd58c00bbf8837b9fd0144924db5e7b1c543d839e';
let outFname = 'tsprotocol.go';
let fda: number, fdb: number, fde: number;  // file descriptors

export function createOutputFiles() {
  fda = fs.openSync('/tmp/ts-a', 'w');  // dump of AST
  fdb = fs.openSync('/tmp/ts-b', 'w');  // unused, for debugging
  fde = fs.openSync(outFname, 'w');     // generated Go
}
export function pra(s: string) {
  return (fs.writeSync(fda, s));
}
export function prb(s: string) {
  return (fs.writeSync(fdb, s));
}
export function prgo(s: string) {
  return (fs.writeSync(fde, s));
}

// Get the hash value of the git commit
export function git(): string {
  let a = fs.readFileSync(`${dir}${srcDir}/.git/HEAD`).toString();
  // ref: refs/heads/foo, or a hash like
  // cc12d1a1c7df935012cdef5d085cdba04a7c8ebe
  if (a.charAt(a.length - 1) == '\n') {
    a = a.substring(0, a.length - 1);
  }
  if (a.length == 40) {
    return a;  // a hash
  }
  if (a.substring(0, 5) == 'ref: ') {
    const fname = `${dir}${srcDir}/.git/` + a.substring(5);
    let b = fs.readFileSync(fname).toString();
    if (b.length == 41) {
      return b.substring(0, 40);
    }
  }
  throw new Error('failed to find the git commit hash');
}

// Produce a header for Go output files
export function computeHeader(pkgDoc: boolean): string {
  let lastMod = 0;
  let lastDate = new Date();
  for (const f of fnames) {
    const st = fs.statSync(f);
    if (st.mtimeMs > lastMod) {
      lastMod = st.mtimeMs;
      lastDate = st.mtime;
    }
  }
  const cp = `// Copyright 2019 The Go Authors. All rights reserved.
  // Use of this source code is governed by a BSD-style
  // license that can be found in the LICENSE file.

  `;
  const a =
    '// Package protocol contains data types and code for LSP jsonrpcs\n' +
    '// generated automatically from vscode-languageserver-node\n' +
    `// commit: ${gitHash}\n` +
    `// last fetched ${lastDate}\n`;
  const b = 'package protocol\n';
  const c = '\n// Code generated (see typescript/README.md) DO NOT EDIT.\n\n';
  if (pkgDoc) {
    return cp + a + b + c;
  }
  else {
    return cp + b + a + c;
  }
}

// Turn a typescript name into an exportable Go name, and appease lint
export function goName(s: string): string {
  let ans = s;
  if (s.charAt(0) == '_') {
    // in the end, none of these are emitted.
    ans = 'Inner' + s.substring(1);
  }
  else { ans = s.substring(0, 1).toUpperCase() + s.substring(1); }
  ans = ans.replace(/Uri$/, 'URI');
  ans = ans.replace(/Id$/, 'ID');
  return ans;
}

// Generate JSON tag for a struct field
export function JSON(n: ts.PropertySignature): string {
  const json = `\`json:"${n.name.getText()}${n.questionToken !== undefined ? ',omitempty' : ''}"\``;
  return json;
}

// Generate modifying prefixes and suffixes to ensure
// consts are unique. (Go consts are package-level, but Typescript's are
// not.) Use suffixes to minimize changes to gopls.
export function constName(nm: string, type: string): string {
  let pref = new Map<string, string>([
    ['DiagnosticSeverity', 'Severity'], ['WatchKind', 'Watch'],
    ['SignatureHelpTriggerKind', 'Sig'], ['CompletionItemTag', 'Compl'],
    ['Integer', 'INT_'], ['Uinteger', 'UINT_']
  ]);  // typeName->prefix
  let suff = new Map<string, string>([
    ['CompletionItemKind', 'Completion'], ['InsertTextFormat', 'TextFormat'],
    ['SymbolTag', 'Symbol'], ['FileOperationPatternKind', 'Op'],
  ]);
  let ans = nm;
  if (pref.get(type)) ans = pref.get(type) + ans;
  if (suff.has(type)) ans = ans + suff.get(type);
  return ans;
}

// Find the comments associated with an AST node
export function getComments(node: ts.Node): string {
  const sf = node.getSourceFile();
  const start = node.getStart(sf, false);
  const starta = node.getStart(sf, true);
  const x = sf.text.substring(starta, start);
  return x;
}


// --------- printing the AST, for debugging

export function printAST(program: ts.Program) {
  // dump the ast, for debugging
  const f = function (n: ts.Node) {
    describe(n, pra);
  };
  for (const sourceFile of program.getSourceFiles()) {
    if (!sourceFile.isDeclarationFile) {
      // walk the tree to do stuff
      ts.forEachChild(sourceFile, f);
    }
  }
  pra('\n');
  for (const key of Object.keys(seenThings).sort()) {
    pra(`${key}: ${seenThings.get(key)} \n`);
  }
}

// Used in printing the AST
let seenThings = new Map<string, number>();
function seenAdd(x: string) {
  const u = seenThings.get(x);
  seenThings.set(x, u === undefined ? 1 : u + 1);
}

// eslint-disable-next-line no-unused-vars
function describe(node: ts.Node, pr: (_: string) => any) {
  if (node === undefined) {
    return;
  }
  let indent = '';

  function f(n: ts.Node) {
    seenAdd(kinds(n));
    if (ts.isIdentifier(n)) {
      pr(`${indent} ${loc(n)} ${strKind(n)} ${n.text} \n`);
    }
    else if (ts.isPropertySignature(n) || ts.isEnumMember(n)) {
      pra(`${indent} ${loc(n)} ${strKind(n)} \n`);
    }
    else if (ts.isTypeLiteralNode(n)) {
      let m = n.members;
      pr(`${indent} ${loc(n)} ${strKind(n)} ${m.length} \n`);
    }
    else if (ts.isStringLiteral(n)) {
      pr(`${indent} ${loc(n)} ${strKind(n)} ${n.text} \n`);
    }
    else { pr(`${indent} ${loc(n)} ${strKind(n)} \n`); }
    indent += ' .';
    ts.forEachChild(n, f);
    indent = indent.slice(0, indent.length - 2);
  }
  f(node);
}


// For debugging, say where an AST node is in a file
export function loc(node: ts.Node | undefined): string {
  if (!node) throw new Error('loc called with undefined (cannot happen!)');
  const sf = node.getSourceFile();
  const start = node.getStart();
  const x = sf.getLineAndCharacterOfPosition(start);
  const full = node.getFullStart();
  const y = sf.getLineAndCharacterOfPosition(full);
  let fn = sf.fileName;
  const n = fn.search(/-node./);
  fn = fn.substring(n + 6);
  return `${fn} ${x.line + 1}: ${x.character + 1} (${y.line + 1}: ${y.character + 1})`;
}

// --- various string stuff

// return a string of the kinds of the immediate descendants
// as part of printing the AST tree
function kinds(n: ts.Node): string {
  let res = 'Seen ' + strKind(n);
  function f(n: ts.Node): void { res += ' ' + strKind(n); }
  ts.forEachChild(n, f);
  return res;
}

// What kind of AST node is it? This would just be typescript's
// SyntaxKind[n.kind] except that the default names for some nodes
// are misleading
export function strKind(n: ts.Node | undefined): string {
  if (n == null || n == undefined) {
    return 'null';
  }
  return kindToStr(n.kind);
}

function kindToStr(k: ts.SyntaxKind): string {
  const x = ts.SyntaxKind[k];
  // some of these have two names
  switch (x) {
    default:
      return x;
    case 'FirstAssignment':
      return 'EqualsToken';
    case 'FirstBinaryOperator':
      return 'LessThanToken';
    case 'FirstCompoundAssignment':
      return 'PlusEqualsToken';
    case 'FirstContextualKeyword':
      return 'AbstractKeyword';
    case 'FirstLiteralToken':
      return 'NumericLiteral';
    case 'FirstNode':
      return 'QualifiedName';
    case 'FirstTemplateToken':
      return 'NoSubstitutionTemplateLiteral';
    case 'LastTemplateToken':
      return 'TemplateTail';
    case 'FirstTypeNode':
      return 'TypePredicate';
  }
}
