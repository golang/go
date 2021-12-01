/* eslint-disable no-useless-return */
// read files from vscode-languageserver-node, and generate Go rpc stubs
// and data definitions. (and maybe someday unmarshaling code)

// The output is 3 files, tsprotocol.go contains the type definitions
// while tsclient.go and tsserver.go contain the LSP API and stub. An LSP server
// uses both APIs. To read the code, start in this file's main() function.

// The code is rich in heuristics and special cases, some of which are to avoid
// extensive changes to gopls, and some of which are due to the mismatch between
// typescript and Go types. In particular, there is no Go equivalent to union
// types, so each case ought to be considered separately. The Go equivalent of A
// & B could frequently be struct{A;B;}, or it could be the equivalent type
// listing all the members of A and B. Typically the code uses the former, but
// especially if A and B have elements with the same name, it does a version of
// the latter. ClientCapabilities has to be expanded, and ServerCapabilities is
// expanded to make the generated code easier to read.

// for us typescript ignorati, having an import makes this file a module
import * as fs from 'fs';
import * as ts from 'typescript';
import * as u from './util';
import { constName, getComments, goName, loc, strKind } from './util';

var program: ts.Program;

function parse() {
  // this won't complain if some fnames don't exist
  program = ts.createProgram(
    u.fnames,
    { target: ts.ScriptTarget.ES2018, module: ts.ModuleKind.CommonJS });
  program.getTypeChecker();  // finish type checking and assignment
}

// ----- collecting information for RPCs
let req = new Map<string, ts.NewExpression>();               // requests
let not = new Map<string, ts.NewExpression>();               // notifications
let ptypes = new Map<string, [ts.TypeNode, ts.TypeNode]>();  // req, resp types
let receives = new Map<string, 'server' | 'client'>();         // who receives it
let rpcTypes = new Set<string>();  // types seen in the rpcs

function findRPCs(node: ts.Node) {
  if (!ts.isModuleDeclaration(node)) {
    return;
  }
  if (!ts.isIdentifier(node.name)) {
    throw new Error(
      `expected Identifier, got ${strKind(node.name)} at ${loc(node)}`);
  }
  let reqnot = req;
  let v = node.name.getText();
  if (v.endsWith('Notification')) reqnot = not;
  else if (!v.endsWith('Request')) return;

  if (!ts.isModuleBlock(node.body)) {
    throw new Error(
      `expected ModuleBlock got ${strKind(node.body)} at ${loc(node)}`);
  }
  let x: ts.ModuleBlock = node.body;
  // The story is to expect const method = 'textDocument/implementation'
  // const type = new ProtocolRequestType<...>(method)
  // but the method may be an explicit string
  let rpc: string = '';
  let newNode: ts.NewExpression;
  for (let i = 0; i < x.statements.length; i++) {
    const uu = x.statements[i];
    if (!ts.isVariableStatement(uu)) continue;
    const dl: ts.VariableDeclarationList = uu.declarationList;
    if (dl.declarations.length != 1)
      throw new Error(`expected a single decl at ${loc(dl)}`);
    const decl: ts.VariableDeclaration = dl.declarations[0];
    const name = decl.name.getText();
    // we want the initializers
    if (name == 'method') {  // mostly StringLiteral but NoSubstitutionTemplateLiteral in protocol.semanticTokens.ts
      if (!ts.isStringLiteral(decl.initializer)) {
        if (!ts.isNoSubstitutionTemplateLiteral(decl.initializer)) {
          console.log(`81: ${decl.initializer.getText()}`);
          throw new Error(`expect StringLiteral at ${loc(decl)} got ${strKind(decl.initializer)}`);
        }
      }
      rpc = decl.initializer.getText();
    }
    else if (name == 'type') {  // NewExpression
      if (!ts.isNewExpression(decl.initializer))
        throw new Error(`89 expected new at ${loc(decl)}`);
      const nn: ts.NewExpression = decl.initializer;
      newNode = nn;
      const mtd = nn.arguments[0];
      if (ts.isStringLiteral(mtd)) rpc = mtd.getText();
      switch (nn.typeArguments.length) {
        case 1:  // exit
          ptypes.set(rpc, [nn.typeArguments[0], null]);
          break;
        case 2:  // notifications
          ptypes.set(rpc, [nn.typeArguments[0], null]);
          break;
        case 4:  // request with no parameters
          ptypes.set(rpc, [null, nn.typeArguments[0]]);
          break;
        case 5:  // request req, resp, partial(?)
          ptypes.set(rpc, [nn.typeArguments[0], nn.typeArguments[1]]);
          break;
        default:
          throw new Error(`${nn.typeArguments?.length} at ${loc(nn)}`);
      }
    }
  }
  if (rpc == '') throw new Error(`112 no name found at ${loc(x)}`);
  // remember the implied types
  const [a, b] = ptypes.get(rpc);
  const add = function (n: ts.Node) {
    rpcTypes.add(goName(n.getText()));
  };
  underlying(a, add);
  underlying(b, add);
  rpc = rpc.substring(1, rpc.length - 1);  // 'exit'
  reqnot.set(rpc, newNode);
}

function setReceives() {
  // mark them all as server, then adjust the client ones.
  // it would be nice to have some independent check on this
  // (this logic fails if the server ever sends $/canceRequest
  //  or $/progress)
  req.forEach((_, k) => { receives.set(k, 'server'); });
  not.forEach((_, k) => { receives.set(k, 'server'); });
  receives.set('window/showMessage', 'client');
  receives.set('window/showMessageRequest', 'client');
  receives.set('window/logMessage', 'client');
  receives.set('telemetry/event', 'client');
  receives.set('client/registerCapability', 'client');
  receives.set('client/unregisterCapability', 'client');
  receives.set('workspace/workspaceFolders', 'client');
  receives.set('workspace/configuration', 'client');
  receives.set('workspace/applyEdit', 'client');
  receives.set('textDocument/publishDiagnostics', 'client');
  receives.set('window/workDoneProgress/create', 'client');
  receives.set('window/showDocument', 'client');
  receives.set('$/progress', 'client');
  // a small check
  receives.forEach((_, k) => {
    if (!req.get(k) && !not.get(k)) throw new Error(`145 missing ${k}}`);
    if (req.get(k) && not.get(k)) throw new Error(`146 dup ${k}`);
  });
}

type DataKind = 'module' | 'interface' | 'alias' | 'enum' | 'class';

interface Data {
  kind: DataKind;
  me: ts.Node;   // root node for this type
  name: string;  // Go name
  origname: string; // their name
  generics: ts.NodeArray<ts.TypeParameterDeclaration>;
  as: ts.NodeArray<ts.HeritageClause>;  // inheritance
  // Interface
  properties: ts.NodeArray<ts.PropertySignature>
  alias: ts.TypeNode;                        // type alias
  // module
  statements: ts.NodeArray<ts.Statement>;
  enums: ts.NodeArray<ts.EnumMember>;
  // class
  members: ts.NodeArray<ts.PropertyDeclaration>;
}
function newData(n: ts.Node, nm: string, k: DataKind, origname: string): Data {
  return {
    kind: k,
    me: n, name: goName(nm), origname: origname,
    generics: ts.factory.createNodeArray<ts.TypeParameterDeclaration>(),
    as: ts.factory.createNodeArray<ts.HeritageClause>(),
    properties: ts.factory.createNodeArray<ts.PropertySignature>(), alias: undefined,
    statements: ts.factory.createNodeArray<ts.Statement>(),
    enums: ts.factory.createNodeArray<ts.EnumMember>(),
    members: ts.factory.createNodeArray<ts.PropertyDeclaration>(),
  };
}

// for debugging, produce a skeleton description
function strData(d: Data): string {
  if (!d) { return 'nil'; }
  const f = function (na: ts.NodeArray<any>): number {
    return na.length;
  };
  const nm = d.name == d.origname ? `${d.name}` : `${d.name}/${d.origname}`;
  return `g:${f(d.generics)} a:${f(d.as)} p:${f(d.properties)} s:${f(d.statements)} e:${f(d.enums)} m:${f(d.members)} a:${d.alias !== undefined} D(${nm}) k:${d.kind}`;
}

let data = new Map<string, Data>();            // parsed data types
let seenTypes = new Map<string, Data>();       // type names we've seen
let extraTypes = new Map<string, string[]>();  // to avoid struct params

function setData(nm: string, d: Data) {
  const v = data.get(nm);
  if (!v) {
    data.set(nm, d);
    return;
  }
  // if there are multiple definitions of the same name, decide what to do.
  // For now the choices are only aliases and modules
  // alias is preferred unless the constant values are needed
  if (nm === 'PrepareSupportDefaultBehavior') {
    // want the alias, as we're going to change the type and can't afford a constant
    if (d.kind === 'alias') data.set(nm, d);
    else if (v.kind == 'alias') data.set(nm, v);
    else throw new Error(`208 ${d.kind} ${v.kind}`);
    return;
  }
  if (nm === 'CodeActionKind') {
    // want the module, need the constants
    if (d.kind === 'module') data.set(nm, d);
    else if (v.kind === 'module') data.set(nm, v);
    else throw new Error(`215 ${d.kind} ${v.kind}`);
  }
  if (v.kind === 'alias' && d.kind !== 'alias') return;
  if (d.kind === 'alias' && v.kind !== 'alias') {
    data.set(nm, d);
    return;
  }
  if (v.kind === 'alias' && d.kind === 'alias') return;
  // protocol/src/common/protocol.foldingRange.ts 44: 1 (39: 2) and
  // types/src/main.ts 397: 1 (392: 2)
  // for FoldingRangeKind
  if (d.me.getText() === v.me.getText()) return;
  // error messages for an unexpected case
  console.log(`228 ${strData(v)} ${loc(v.me)} for`);
  console.log(`229 ${v.me.getText().replace(/\n/g, '\\n')}`);
  console.log(`230 ${strData(d)} ${loc(d.me)}`);
  console.log(`231 ${d.me.getText().replace(/\n/g, '\\n')}`);
  throw new Error(`232 setData found ${v.kind} for ${d.kind}`);
}

// look at top level data definitions
function genTypes(node: ts.Node) {
  // Ignore top-level items that can't produce output
  if (ts.isExpressionStatement(node) || ts.isFunctionDeclaration(node) ||
    ts.isImportDeclaration(node) || ts.isVariableStatement(node) ||
    ts.isExportDeclaration(node) || ts.isEmptyStatement(node) ||
    ts.isExportAssignment(node) || ts.isImportEqualsDeclaration(node) ||
    ts.isBlock(node) || node.kind == ts.SyntaxKind.EndOfFileToken) {
    return;
  }
  if (ts.isInterfaceDeclaration(node)) {
    const v: ts.InterfaceDeclaration = node;
    // need to check the members, many of which are disruptive
    let mems: ts.PropertySignature[] = [];
    const f = function (t: ts.TypeElement) {
      if (ts.isPropertySignature(t)) {
        mems.push(t);
      } else if (ts.isMethodSignature(t) || ts.isCallSignatureDeclaration(t)) {
        return;
      } else if (ts.isIndexSignatureDeclaration(t)) {
        // probably safe to ignore these
        // [key: string]: boolean | number | string | undefined;
        // and InitializeResult: [custom: string]: any;]
      } else
        throw new Error(`259 unexpected ${strKind(t)}`);
    };
    v.members.forEach(f);
    if (mems.length == 0 && !v.heritageClauses &&
      v.name.getText() != 'InitializedParams') {
      return;  // Don't seem to need any of these [Logger, PipTransport, ...]
    }
    // Found one we want
    let x = newData(v, goName(v.name.getText()), 'interface', v.name.getText());
    x.properties = ts.factory.createNodeArray<ts.PropertySignature>(mems);
    if (v.typeParameters) x.generics = v.typeParameters;
    if (v.heritageClauses) x.as = v.heritageClauses;
    if (x.generics.length > 1) {  // Unneeded
      // Item interface Item<K, V>...
      return;
    }
    if (data.has(x.name)) {  // modifying one we've seen
      x = dataChoose(x, data.get(x.name));
    }
    setData(x.name, x);
  } else if (ts.isTypeAliasDeclaration(node)) {
    const v: ts.TypeAliasDeclaration = node;
    let x = newData(v, v.name.getText(), 'alias', v.name.getText());
    x.alias = v.type;
    // if type is a union of constants, we (mostly) don't want it
    // (at the top level)
    // Unfortunately this is false for TraceValues
    if (ts.isUnionTypeNode(v.type) &&
      v.type.types.every((n: ts.TypeNode) => ts.isLiteralTypeNode(n))) {
      if (x.name != 'TraceValues') return;
    }
    if (v.typeParameters) {
      x.generics = v.typeParameters;
    }
    if (data.has(x.name)) x = dataChoose(x, data.get(x.name));
    if (x.generics.length > 1) {
      return;
    }
    setData(x.name, x);
  } else if (ts.isModuleDeclaration(node)) {
    const v: ts.ModuleDeclaration = node;
    if (!ts.isModuleBlock(v.body)) {
      throw new Error(`${loc(v)} not ModuleBlock, but ${strKind(v.body)}`);
    }
    const b: ts.ModuleBlock = v.body;
    var s: ts.Statement[] = [];
    // we don't want most of these
    const fx = function (x: ts.Statement) {
      if (ts.isFunctionDeclaration(x)) {
        return;
      }
      if (ts.isTypeAliasDeclaration(x) || ts.isModuleDeclaration(x)) {
        return;
      }
      if (!ts.isVariableStatement(x))
        throw new Error(
          `315 expected VariableStatment ${loc(x)} ${strKind(x)} ${x.getText()}`);
      if (hasNewExpression(x)) {
        return;
      }
      s.push(x);
    };
    b.statements.forEach(fx);
    if (s.length == 0) {
      return;
    }
    let m = newData(node, v.name.getText(), 'module', v.name.getText());
    m.statements = ts.factory.createNodeArray<ts.Statement>(s);
    if (data.has(m.name)) m = dataChoose(m, data.get(m.name));
    setData(m.name, m);
  } else if (ts.isEnumDeclaration(node)) {
    const nm = node.name.getText();
    let v = newData(node, nm, 'enum', node.name.getText());
    v.enums = node.members;
    if (data.has(nm)) {
      v = dataChoose(v, data.get(nm));
    }
    setData(nm, v);
  } else if (ts.isClassDeclaration(node)) {
    const v: ts.ClassDeclaration = node;
    var d: ts.PropertyDeclaration[] = [];
    const wanted = function (c: ts.ClassElement): string {
      if (ts.isConstructorDeclaration(c)) {
        return '';
      }
      if (ts.isMethodDeclaration(c)) {
        return '';
      }
      if (ts.isGetAccessor(c)) {
        return '';
      }
      if (ts.isSetAccessor(c)) {
        return '';
      }
      if (ts.isPropertyDeclaration(c)) {
        d.push(c);
        return strKind(c);
      }
      throw new Error(`Class decl ${strKind(c)} `);
    };
    v.members.forEach((c) => wanted(c));
    if (d.length == 0) {
      return;
    }  // don't need it
    let c = newData(v, v.name.getText(), 'class', v.name.getText());
    c.members = ts.factory.createNodeArray<ts.PropertyDeclaration>(d);
    if (v.typeParameters) {
      c.generics = v.typeParameters;
    }
    if (c.generics.length > 1) {
      return;
    }
    if (v.heritageClauses) {
      c.as = v.heritageClauses;
    }
    if (data.has(c.name))
      throw new Error(`Class dup ${loc(c.me)} and ${loc(data.get(c.name).me)}`);
    setData(c.name, c);
  } else {
    throw new Error(`378 unexpected ${strKind(node)} ${loc(node)} `);
  }
}

// Typescript can accumulate, but this chooses one or the other
function dataChoose(a: Data, b: Data): Data {
  // maybe they are textually identical? (e.g., FoldingRangeKind)
  const [at, bt] = [a.me.getText(), b.me.getText()];
  if (at == bt) {
    return a;
  }
  switch (a.name) {
    case 'InitializeError':
    case 'CompletionItemTag':
    case 'SymbolTag':
    case 'CodeActionKind':
    case 'Integer':
    case 'Uinteger':
    case 'Decimal':
      // want the Module, if anything
      return a.statements.length > 0 ? a : b;
    case 'CancellationToken':
    case 'CancellationStrategy':
      // want the Interface
      return a.properties.length > 0 ? a : b;
    case 'TextDocumentContentChangeEvent':  // almost the same
    case 'TokenFormat':
    case 'PrepareSupportDefaultBehavior':
      return a;
  }
  console.log(
    `409 ${strKind(a.me)} ${strKind(b.me)} ${a.name} ${loc(a.me)} ${loc(b.me)}`);
  throw new Error(`410 Fix dataChoose for ${a.name}`);
}

// is a node an ancestor of a NewExpression
function hasNewExpression(n: ts.Node): boolean {
  let ans = false;
  n.forEachChild((n: ts.Node) => {
    if (ts.isNewExpression(n)) ans = true;
  });
  return ans;
}

function checkOnce() {
  // Data for all the rpc types?
  rpcTypes.forEach(s => {
    if (!data.has(s)) throw new Error(`checkOnce, ${s}?`);
  });
}

// helper function to find underlying types
// eslint-disable-next-line no-unused-vars
function underlying(n: ts.Node | undefined, f: (n: ts.Node) => void) {
  if (!n) return;
  const ff = function (n: ts.Node) {
    underlying(n, f);
  };
  if (ts.isIdentifier(n)) {
    f(n);
  } else if (
    n.kind == ts.SyntaxKind.StringKeyword ||
    n.kind == ts.SyntaxKind.NumberKeyword ||
    n.kind == ts.SyntaxKind.AnyKeyword ||
    n.kind == ts.SyntaxKind.UnknownKeyword ||
    n.kind == ts.SyntaxKind.NullKeyword ||
    n.kind == ts.SyntaxKind.BooleanKeyword ||
    n.kind == ts.SyntaxKind.ObjectKeyword ||
    n.kind == ts.SyntaxKind.VoidKeyword) {
    // nothing to do
  } else if (ts.isTypeReferenceNode(n)) {
    f(n.typeName);
  } else if (ts.isArrayTypeNode(n)) {
    underlying(n.elementType, f);
  } else if (ts.isHeritageClause(n)) {
    n.types.forEach(ff);
  } else if (ts.isExpressionWithTypeArguments(n)) {
    underlying(n.expression, f);
  } else if (ts.isPropertySignature(n)) {
    underlying(n.type, f);
  } else if (ts.isTypeLiteralNode(n)) {
    n.members.forEach(ff);
  } else if (ts.isUnionTypeNode(n) || ts.isIntersectionTypeNode(n)) {
    n.types.forEach(ff);
  } else if (ts.isIndexSignatureDeclaration(n)) {
    underlying(n.type, f);
  } else if (ts.isParenthesizedTypeNode(n)) {
    underlying(n.type, f);
  } else if (
    ts.isLiteralTypeNode(n) || ts.isVariableStatement(n) ||
    ts.isTupleTypeNode(n)) {
    // we only see these in moreTypes, but they are handled elsewhere
  } else if (ts.isEnumMember(n)) {
    if (ts.isStringLiteral(n.initializer)) return;
    throw new Error(`472 EnumMember ${strKind(n.initializer)} ${n.name.getText()}`);
  } else {
    throw new Error(`474 saw ${strKind(n)} in underlying. ${n.getText()} at ${loc(n)}`);
  }
}

// find all the types implied by seenTypes.
// Simplest way to the transitive closure is to stabilize the size of seenTypes
// but it is slow
function moreTypes() {
  const extra = function (s: string) {
    if (!data.has(s)) throw new Error(`moreTypes needs ${s}`);
    seenTypes.set(s, data.get(s));
  };
  rpcTypes.forEach(extra);  // all the types needed by the rpcs
  // needed in enums.go (or elsewhere)
  extra('InitializeError');
  extra('WatchKind');
  extra('FoldingRangeKind');
  // not sure why these weren't picked up
  extra('DidChangeWatchedFilesRegistrationOptions');
  extra('WorkDoneProgressBegin');
  extra('WorkDoneProgressReport');
  extra('WorkDoneProgressEnd');
  let old = 0;
  do {
    old = seenTypes.size;

    const m = new Map<string, Data>();
    const add = function (n: ts.Node) {
      const nm = goName(n.getText());
      if (seenTypes.has(nm) || m.has(nm)) return;
      if (data.get(nm)) {
        m.set(nm, data.get(nm));
      }
    };
    // expect all the heritage clauses have single Identifiers
    const h = function (n: ts.Node) {
      underlying(n, add);
    };
    const f = function (x: ts.NodeArray<ts.Node>) {
      x.forEach(h);
    };
    seenTypes.forEach((d: Data) => d && f(d.as));
    // find the types in the properties
    seenTypes.forEach((d: Data) => d && f(d.properties));
    // and in the alias and in the statements and in the enums
    seenTypes.forEach((d: Data) => d && underlying(d.alias, add));
    seenTypes.forEach((d: Data) => d && f(d.statements));
    seenTypes.forEach((d: Data) => d && f(d.enums));
    m.forEach((d, k) => seenTypes.set(k, d));
  }
  while (seenTypes.size != old)
    ;
}

function cleanData() { // middle pass
  // seenTypes contains all the top-level types.
  seenTypes.forEach((d) => {
    if (d.kind == 'alias') mergeAlias(d);
  });
}

function sameType(a: ts.TypeNode, b: ts.TypeNode): boolean {
  if (a.kind !== b.kind) return false;
  if (a.kind === ts.SyntaxKind.BooleanKeyword) return true;
  if (a.kind === ts.SyntaxKind.StringKeyword) return true;
  if (ts.isTypeReferenceNode(a) && ts.isTypeReferenceNode(b) &&
    a.typeName.getText() === b.typeName.getText()) return true;
  if (ts.isArrayTypeNode(a) && ts.isArrayTypeNode(b)) return sameType(a.elementType, b.elementType);
  if (ts.isTypeLiteralNode(a) && ts.isTypeLiteralNode(b)) {
    if (a.members.length !== b.members.length) return false;
    if (a.members.length === 1) return a.members[0].name.getText() === b.members[0].name.getText();
    if (loc(a) === loc(b)) return true;
  }
  throw new Error(`544 sameType? ${strKind(a)} ${strKind(b)} ${a.getText()}`);
}
type CreateMutable<Type> = {
  -readonly [Property in keyof Type]: Type[Property];
};
type propMap = Map<string, ts.PropertySignature>;
function propMapSet(pm: propMap, name: string, v: ts.PropertySignature) {
  if (!pm.get(name)) {
    try { getComments(v); } catch (e) { console.log(`552 ${name} ${e}`); }
    pm.set(name, v);
    return;
  }
  const a = pm.get(name).type;
  const b = v.type;
  if (sameType(a, b)) {
    return;
  }
  if (ts.isTypeReferenceNode(a) && ts.isTypeLiteralNode(b)) {
    const x = mergeTypeRefLit(a, b);
    const fake: CreateMutable<ts.PropertySignature> = v;
    fake['type'] = x;
    check(fake as ts.PropertySignature, '565');
    pm.set(name, fake as ts.PropertySignature);
    return;
  }
  if (ts.isTypeLiteralNode(a) && ts.isTypeLiteralNode(b)) {
    const x = mergeTypeLitLit(a, b);
    const fake: CreateMutable<ts.PropertySignature> = v;
    fake['type'] = x;
    check(fake as ts.PropertySignature, '578');
    pm.set(name, fake as ts.PropertySignature);
    return;
  }
  console.log(`577 ${pm.get(name).getText()}\n${v.getText()}`);
  throw new Error(`578 should merge ${strKind(a)} and ${strKind(b)} for ${name}`);
}
function addToProperties(pm: propMap, tn: ts.TypeNode | undefined, prefix = '') {
  if (!tn) return;
  if (ts.isTypeReferenceNode(tn)) {
    const d = seenTypes.get(goName(tn.typeName.getText()));
    if (tn.typeName.getText() === 'T') return;
    if (!d) throw new Error(`584 ${tn.typeName.getText()} not found`);
    if (d.properties.length === 0 && d.alias === undefined) return;
    if (d.alias !== undefined) {
      if (ts.isIntersectionTypeNode(d.alias)) {
        d.alias.types.forEach((tn) => addToProperties(pm, tn, prefix)); // prefix?
        return;
      }
    }
    d.properties.forEach((ps) => {
      const name = `${prefix}.${ps.name.getText()}`;
      propMapSet(pm, name, ps);
      addToProperties(pm, ps.type, name);
    });
  } else if (strKind(tn) === 'TypeLiteral') {
    if (!ts.isTypeLiteralNode(tn)) new Error(`598 ${strKind(tn)}`);
    tn.forEachChild((child: ts.Node) => {
      if (!ts.isPropertySignature(child)) throw new Error(`600 ${strKind(child)}`);
      const name = `${prefix}.${child.name.getText()}`;
      propMapSet(pm, name, child);
      addToProperties(pm, child.type, name);
    });
  }
}
function deepProperties(d: Data): propMap | undefined {
  let properties: propMap = new Map<string, ts.PropertySignature>();
  if (!d.alias || !ts.isIntersectionTypeNode(d.alias)) return undefined;
  d.alias.types.forEach((ts) => addToProperties(properties, ts));
  return properties;
}

function mergeAlias(d: Data) {
  const props = deepProperties(d);
  if (!props) return; // nothing merged
  // now each element of props should have length 1
  // change d to merged, toss its alias field, fill in its properties
  const v: ts.PropertySignature[] = [];
  props.forEach((ps, nm) => {
    const xlen = nm.split('.').length;
    if (xlen !== 2) return; // not top-level
    v.push(ps);
  });
  d.kind = 'interface';
  d.alias = undefined;
  d.properties = ts.factory.createNodeArray(v);
}

function mergeTypeLitLit(a: ts.TypeLiteralNode, b: ts.TypeLiteralNode): ts.TypeLiteralNode {
  const v = new Map<string, ts.TypeElement>(); // avoid duplicates
  a.members.forEach((te) => v.set(te.name.getText(), te));
  b.members.forEach((te) => v.set(te.name.getText(), te));
  const x: ts.TypeElement[] = [];
  v.forEach((te) => x.push(te));
  const fake: CreateMutable<ts.TypeLiteralNode> = a;
  fake['members'] = ts.factory.createNodeArray(x);
  check(fake as ts.TypeLiteralNode, '643');
  return fake as ts.TypeLiteralNode;
}

function mergeTypeRefLit(a: ts.TypeReferenceNode, b: ts.TypeLiteralNode): ts.TypeLiteralNode {
  const d = seenTypes.get(goName(a.typeName.getText()));
  if (!d) throw new Error(`644 name ${a.typeName.getText()} not found`);
  const typ = d.me;
  if (!ts.isInterfaceDeclaration(typ)) throw new Error(`646 got ${strKind(typ)} not InterfaceDecl`);
  const v = new Map<string, ts.TypeElement>(); // avoid duplicates
  typ.members.forEach((te) => v.set(te.name.getText(), te));
  b.members.forEach((te) => v.set(te.name.getText(), te));
  const x: ts.TypeElement[] = [];
  v.forEach((te) => x.push(te));

  const w = ts.factory.createNodeArray(x);
  const fk: CreateMutable<ts.TypeLiteralNode> = b;
  fk['members'] = w;
  (fk['members'] as { pos: number })['pos'] = b.members.pos;
  (fk['members'] as { end: number })['end'] = b.members.end;
  check(fk as ts.TypeLiteralNode, '662');
  return fk as ts.TypeLiteralNode;
}

// check that constructed nodes still have associated text
function check(n: ts.Node, loc: string) {
  try { getComments(n); } catch (e) { console.log(`check at ${loc} ${e}`); }
  try { n.getText(); } catch (e) { console.log(`text check at ${loc}`); }
}

let typesOut = new Array<string>();
let constsOut = new Array<string>();

// generate Go types
function toGo(d: Data, nm: string) {
  if (!d) return;  // this is probably a generic T
  if (d.name.startsWith('Inner') || d.name === 'WindowClientCapabilities') return; // removed by alias processing
  if (d.name === 'Integer' || d.name === 'Uinteger') return; // unneeded
  switch (d.kind) {
    case 'alias':
      goTypeAlias(d, nm); break;
    case 'module': goModule(d, nm); break;
    case 'enum': goEnum(d, nm); break;
    case 'interface': goInterface(d, nm); break;
    default:
      throw new Error(
        `672: more cases in toGo ${nm} ${d.kind}`);
  }
}

// these fields need a * and are not covered by the code
// that calls isStructType.
var starred: [string, string][] = [
  ['TextDocumentContentChangeEvent', 'range'], ['CodeAction', 'command'],
  ['CodeAction', 'disabled'],
  ['DidSaveTextDocumentParams', 'text'], ['CompletionItem', 'command'],
  ['Diagnostic', 'codeDescription']
];

// generate Go code for an interface
function goInterface(d: Data, nm: string) {
  let ans = `type ${goName(nm)} struct {\n`;

  // generate the code for each member
  const g = function (n: ts.PropertySignature) {
    if (!ts.isPropertySignature(n))
      throw new Error(`expected PropertySignature got ${strKind(n)} `);
    ans = ans.concat(getComments(n));
    const json = u.JSON(n);
    let gt = goType(n.type, n.name.getText());
    if (gt == d.name) gt = '*' + gt; // avoid recursive types (SelectionRange)
    // there are several cases where a * is needed
    // (putting * in front of too many things breaks uses of CodeActionKind)
    starred.forEach(([a, b]) => {
      if (d.name == a && n.name.getText() == b) {
        gt = '*' + gt;
      }
    });
    ans = ans.concat(`${goName(n.name.getText())} ${gt}`, json, '\n');
  };
  d.properties.forEach(g);
  // heritage clauses become embedded types
  // check they are all Identifiers
  const f = function (n: ts.ExpressionWithTypeArguments) {
    if (!ts.isIdentifier(n.expression))
      throw new Error(`Interface ${nm} heritage ${strKind(n.expression)} `);
    if (n.expression.getText() === 'Omit') return;  // Type modification type
    ans = ans.concat(goName(n.expression.getText()), '\n');
  };
  d.as.forEach((n: ts.HeritageClause) => n.types.forEach(f));
  ans = ans.concat('}\n');
  typesOut.push(getComments(d.me));
  typesOut.push(ans);
}

// generate Go code for a module (const declarations)
// Generates type definitions, and named constants
function goModule(d: Data, nm: string) {
  if (d.generics.length > 0 || d.as.length > 0) {
    throw new Error(`743 goModule: unexpected for ${nm}
  `);
  }
  // all the statements should be export const <id>: value
  //   or value = value
  // They are VariableStatements with x.declarationList having a single
  //   VariableDeclaration
  let isNumeric = false;
  const f = function (n: ts.Statement, i: number) {
    if (!ts.isVariableStatement(n)) {
      throw new Error(`753 ${nm} ${i} expected VariableStatement,
      got ${strKind(n)}`);
    }
    const c = getComments(n);
    const v = n.declarationList.declarations[0];  // only one

    if (!v.initializer)
      throw new Error(`760 no initializer ${nm} ${i} ${v.name.getText()}`);
    isNumeric = strKind(v.initializer) == 'NumericLiteral';
    if (c != '') constsOut.push(c);  // no point if there are no comments
    // There are duplicates.
    const cname = constName(goName(v.name.getText()), nm);
    let val = v.initializer.getText();
    val = val.split('\'').join('"');  // useless work for numbers
    constsOut.push(`${cname} ${nm} = ${val}`);
  };
  d.statements.forEach(f);
  typesOut.push(getComments(d.me));
  // Or should they be type aliases?
  typesOut.push(`type ${nm} ${isNumeric ? 'float64' : 'string'}`);
}

// generate Go code for an enum. Both types and named constants
function goEnum(d: Data, nm: string) {
  let isNumeric = false;
  const f = function (v: ts.EnumMember, j: number) {  // same as goModule
    if (!v.initializer)
      throw new Error(`goEnum no initializer ${nm} ${j} ${v.name.getText()}`);
    isNumeric = strKind(v.initializer) == 'NumericLiteral';
    const c = getComments(v);
    const cname = constName(goName(v.name.getText()), nm);
    let val = v.initializer.getText();
    val = val.split('\'').join('"');  // replace quotes. useless work for numbers
    constsOut.push(`${c}${cname} ${nm} = ${val}`);
  };
  d.enums.forEach(f);
  typesOut.push(getComments(d.me));
  // Or should they be type aliases?
  typesOut.push(`type ${nm} ${isNumeric ? 'float64' : 'string'}`);
}

// generate code for a type alias
function goTypeAlias(d: Data, nm: string) {
  if (d.as.length != 0 || d.generics.length != 0) {
    if (nm != 'ServerCapabilities')
      throw new Error(`${nm} has extra fields(${d.as.length},${d.generics.length}) ${d.me.getText()}`);
  }
  typesOut.push(getComments(d.me));
  // d.alias doesn't seem to have comments
  let aliasStr = goName(nm) == 'DocumentURI' ? ' ' : ' = ';
  if (nm == 'PrepareSupportDefaultBehavior') {
    // code-insiders is sending a bool, not a number. PJW: check this after Feb/2021
    // (and gopls never looks at it anyway)
    typesOut.push(`type ${goName(nm)}${aliasStr}interface{}\n`);
    return;
  }
  typesOut.push(`type ${goName(nm)}${aliasStr}${goType(d.alias, nm)}\n`);
}

// return a go type and maybe an assocated javascript tag
function goType(n: ts.TypeNode | undefined, nm: string): string {
  if (!n) throw new Error(`goType undefined for ${nm}`);
  if (n.getText() == 'T') return 'interface{}';  // should check it's generic
  if (ts.isTypeReferenceNode(n)) {
    // DocumentDiagnosticReportKind.unChanged (or .new) value is "new" or "unChanged"
    if (n.getText().startsWith('DocumentDiagnostic')) return 'string';
    switch (n.getText()) {
      case 'integer': return 'int32';
      case 'uinteger': return 'uint32';
      default: return goName(n.typeName.getText());  // avoid <T>
    }
  } else if (ts.isUnionTypeNode(n)) {
    return goUnionType(n, nm);
  } else if (ts.isIntersectionTypeNode(n)) {
    return goIntersectionType(n, nm);
  } else if (strKind(n) == 'StringKeyword') {
    return 'string';
  } else if (strKind(n) == 'NumberKeyword') {
    return 'float64';
  } else if (strKind(n) == 'BooleanKeyword') {
    return 'bool';
  } else if (strKind(n) == 'AnyKeyword' || strKind(n) == 'UnknownKeyword') {
    return 'interface{}';
  } else if (strKind(n) == 'NullKeyword') {
    return 'nil';
  } else if (strKind(n) == 'VoidKeyword' || strKind(n) == 'NeverKeyword') {
    return 'void';
  } else if (strKind(n) == 'ObjectKeyword') {
    return 'interface{}';
  } else if (ts.isArrayTypeNode(n)) {
    if (nm === 'arguments') {
      // Command and ExecuteCommandParams
      return '[]json.RawMessage';
    }
    return `[]${goType(n.elementType, nm)}`;
  } else if (ts.isParenthesizedTypeNode(n)) {
    return goType(n.type, nm);
  } else if (ts.isLiteralTypeNode(n)) {
    return strKind(n.literal) == 'StringLiteral' ? 'string' : 'float64';
  } else if (ts.isTypeLiteralNode(n)) {
    // these are anonymous structs
    const v = goTypeLiteral(n, nm);
    return v;
  } else if (ts.isTupleTypeNode(n)) {
    if (n.getText() == '[number, number]') return '[]float64';
    throw new Error(`goType unexpected Tuple ${n.getText()}`);
  }
  throw new Error(`${strKind(n)} goType unexpected ${n.getText()} for ${nm}`);
}

// The choice is uniform interface{}, or some heuristically assigned choice,
// or some better sytematic idea I haven't thought of. Using interface{}
// is, in practice, impossibly complex in the existing code.
function goUnionType(n: ts.UnionTypeNode, nm: string): string {
  let help = `/*${n.getText()}*/`;  // show the original as a comment
  // There are some bad cases with newlines:
  // range?: boolean | {\n	};
  // full?: boolean | {\n		/**\n		 * The server supports deltas for full documents.\n		 */\n		delta?: boolean;\n	}
  // These are handled specially:
  if (nm == 'range') help = help.replace(/\n/, '');
  if (nm == 'full' && help.indexOf('\n') != -1) {
    help = '/*boolean | <elided struct>*/';
  }
  // handle all the special cases
  switch (n.types.length) {
    case 2: {
      const a = strKind(n.types[0]);
      const b = strKind(n.types[1]);
      if (a == 'NumberKeyword' && b == 'StringKeyword') {  // ID
        return `interface{} ${help}`;
      }
      // for null, b is not useful (LiternalType)
      if (n.types[1].getText() === 'null') {
        if (nm == 'textDocument/codeAction') {
          // (Command | CodeAction)[] | null
          return `[]CodeAction ${help}`;
        }
        let v = goType(n.types[0], 'a');
        return `${v} ${help}`;
      }
      if (a == 'BooleanKeyword') {  // usually want bool
        if (nm == 'codeActionProvider') return `interface{} ${help}`;
        if (nm == 'renameProvider') return `interface{} ${help}`;
        if (nm == 'full') return `interface{} ${help}`; // there's a struct
        if (nm == 'save') return `${goType(n.types[1], '680')} ${help}`;
        return `${goType(n.types[0], 'b')} ${help}`;
      }
      if (b == 'ArrayType') return `${goType(n.types[1], 'c')} ${help}`;
      if (help.includes('InsertReplaceEdit') && n.types[0].getText() == 'TextEdit') {
        return `*TextEdit ${help}`;
      }
      if (a == 'TypeReference') {
        if (nm == 'edits') return `${goType(n.types[0], '901')} ${help}`;
        if (a == b) return `interface{} ${help}`;
        if (nm == 'code') return `interface{} ${help}`;
        if (nm == 'editRange') return `${goType(n.types[0], '904')} ${help}`;
        if (nm === 'location') return `${goType(n.types[0], '905')} ${help}`;
      }
      if (a == 'StringKeyword') return `string ${help}`;
      if (a == 'TypeLiteral' && nm == 'TextDocumentContentChangeEvent') {
        return `${goType(n.types[0], nm)}`;
      }
      if (a == 'TypeLiteral' && b === 'TypeLiteral') {
        // DocumentDiagnosticReport
        // the first one includes the second one
        return `${goType(n.types[0], '9d')}`;
      }
      throw new Error(`911 ${nm}: a:${a}/${goType(n.types[0], '9a')} b:${b}/${goType(n.types[1], '9b')} ${loc(n)}`);
    }
    case 3: {
      const aa = strKind(n.types[0]);
      const bb = strKind(n.types[1]);
      const cc = strKind(n.types[2]);
      if (nm === 'workspace/symbol') return `${goType(n.types[0], '930')} ${help}`;
      if (nm == 'DocumentFilter') {
        // not really a union. the first is enough, up to a missing
        // omitempty but avoid repetitious comments
        return `${goType(n.types[0], 'g')}`;
      }
      if (nm == 'textDocument/documentSymbol') {
        return `[]interface{} ${help}`;
      }
      if (aa == 'TypeReference' && bb == 'ArrayType' && (cc == 'NullKeyword' || cc === 'LiteralType')) {
        return `${goType(n.types[0], 'd')} ${help}`;
      }
      if (aa == 'TypeReference' && bb == aa && cc == 'ArrayType') {
        // should check that this is Hover.Contents
        return `${goType(n.types[0], 'e')} ${help}`;
      }
      if (aa == 'ArrayType' && bb == 'TypeReference' && (cc == 'NullKeyword' || cc === 'LiteralType')) {
        // check this is nm == 'textDocument/completion'
        return `${goType(n.types[1], 'f')} ${help}`;
      }
      if (aa == 'LiteralType' && bb == aa && cc == aa) return `string ${help}`;
      // keep this for diagnosing unexpected interface{} results
      // console.log(`931, interface{} for ${aa}/${goType(n.types[0], 'g')},${bb}/${goType(n.types[1], 'h')},${cc}/${goType(n.types[2], 'i')} ${nm}`);
      break;
    }
    case 4:
      if (nm == 'documentChanges') return `TextDocumentEdit ${help} `;
      if (nm == 'textDocument/prepareRename') return `Range ${help} `;
      break;
    case 8: // LSPany
      break;
    default:
      throw new Error(`957 goUnionType len=${n.types.length} nm=${nm} ${n.getText()}`);
  }

  // Result will be interface{} with a comment
  let isLiteral = true;
  let literal = 'string';
  let res = 'interface{} /* ';
  n.types.forEach((v: ts.TypeNode, i: number) => {
    // might get an interface inside:
    //  (Command | CodeAction)[] | null
    let m = goType(v, nm);
    if (m.indexOf('interface') != -1) {
      // avoid nested comments
      m = m.split(' ')[0];
    }
    m = m.split('\n').join('; ');  // sloppy: struct{;
    res = res.concat(`${i == 0 ? '' : ' | '}`, m);
    if (!ts.isLiteralTypeNode(v)) isLiteral = false;
    else literal = strKind(v.literal) == 'StringLiteral' ? 'string' : 'number';
  });
  if (!isLiteral) {
    return res + '*/';
  }
  // I don't think we get here
  // trace?: 'off' | 'messages' | 'verbose' should get string
  return `${literal} /* ${n.getText()} */`;
}

// some of the intersection types A&B are ok as struct{A;B;} and some
// could be expanded, and ClientCapabilites has to be expanded,
// at least for workspace. It's possible to check algorithmically,
// but much simpler just to check explicitly.
function goIntersectionType(n: ts.IntersectionTypeNode, nm: string): string {
  if (nm == 'ClientCapabilities') return expandIntersection(n);
  //if (nm == 'ServerCapabilities') return expandIntersection(n); // save for later consideration
  let inner = '';
  n.types.forEach(
    (t: ts.TypeNode) => { inner = inner.concat(goType(t, nm), '\n'); });
  return `struct{ \n${inner}} `;
}

// for each of the intersected types, extract its components (each will
// have a Data with properties) extract the properties, and keep track
// of them by name. The names that occur once can be output. The names
// that occur more than once need to be combined.
function expandIntersection(n: ts.IntersectionTypeNode): string {
  const bad = function (n: ts.Node, s: string) {
    return new Error(`expandIntersection ${strKind(n)} ${s}`);
  };
  let props = new Map<string, ts.PropertySignature[]>();
  for (const tp of n.types) {
    if (!ts.isTypeReferenceNode(tp)) throw bad(tp, 'A');
    const d = data.get(goName(tp.typeName.getText()));
    for (const p of d.properties) {
      if (!ts.isPropertySignature(p)) throw bad(p, 'B');
      let v = props.get(p.name.getText()) || [];
      v.push(p);
      props.set(p.name.getText(), v);
    }
  }
  let ans = 'struct {\n';
  for (const [k, v] of Array.from(props)) {
    if (v.length == 1) {
      const a = v[0];
      ans = ans.concat(getComments(a));
      ans = ans.concat(`${goName(k)} ${goType(a.type, k)} ${u.JSON(a)}\n`);
      continue;
    }
    ans = ans.concat(`${goName(k)} struct {\n`);
    for (let i = 0; i < v.length; i++) {
      const a = v[i];
      if (ts.isTypeReferenceNode(a.type)) {
        ans = ans.concat(getComments(a));
        ans = ans.concat(goName(a.type.typeName.getText()), '\n');
      } else if (ts.isTypeLiteralNode(a.type)) {
        if (a.type.members.length != 1) throw bad(a.type, 'C');
        const b = a.type.members[0];
        if (!ts.isPropertySignature(b)) throw bad(b, 'D');
        ans = ans.concat(getComments(b));
        ans = ans.concat(
          goName(b.name.getText()), ' ', goType(b.type, 'a'), u.JSON(b), '\n');
      } else {
        throw bad(a.type, `E ${a.getText()} in ${goName(k)} at ${loc(a)}`);
      }
    }
    ans = ans.concat('}\n');
  }
  ans = ans.concat('}\n');
  return ans;
}

// Does it make sense to use a pointer?
function isStructType(te: ts.TypeNode): boolean {
  switch (strKind(te)) {
    case 'UnionType': // really need to know which type will be chosen
    case 'BooleanKeyword':
    case 'StringKeyword':
    case 'ArrayType':
      return false;
    case 'TypeLiteral': return false; // true makes for difficult compound constants
    // but think more carefully to understands why starred is needed.
    case 'TypeReference': {
      if (!ts.isTypeReferenceNode(te)) throw new Error(`1047 impossible ${strKind(te)}`);
      const d = seenTypes.get(goName(te.typeName.getText()));
      if (d === undefined || d.properties.length == 0) return false;
      if (d.properties.length > 1) return true;
      // alias or interface with a single property (The alias is Uinteger, which we ignore later)
      if (d.alias) return false;
      const x = d.properties[0].type;
      return isStructType(x);
    }
    default: throw new Error(`1055 indirectable> ${strKind(te)}`);
  }
}

function goTypeLiteral(n: ts.TypeLiteralNode, nm: string): string {
  let ans: string[] = [];  // in case we generate a new extra type
  let res = 'struct{\n';   // the actual answer usually
  const g = function (nx: ts.TypeElement) {
    // add the json, as in goInterface(). Strange inside union types.
    if (ts.isPropertySignature(nx)) {
      let json = u.JSON(nx);
      let typ = goType(nx.type, nx.name.getText());
      // }/*\n*/`json:v` is not legal, the comment is a newline
      if (typ.includes('\n') && typ.indexOf('*/') === typ.length - 2) {
        typ = typ.replace(/\n\t*/g, ' ');
      }
      const v = getComments(nx) || '';
      starred.forEach(([a, b]) => {
        if (a != nm || b != typ.toLowerCase()) return;
        typ = '*' + typ;
        json = json.substring(0, json.length - 2) + ',omitempty"`';
      });
      if (typ[0] !== '*' && isStructType(nx.type)) typ = '*' + typ;
      res = res.concat(`${v} ${goName(nx.name.getText())} ${typ}`, json, '\n');
      ans.push(`${v}${goName(nx.name.getText())} ${typ} ${json}\n`);
    } else if (ts.isIndexSignatureDeclaration(nx)) {
      const comment = nx.getText().replace(/[/]/g, '');
      if (nx.getText() == '[uri: string]: TextEdit[];') {
        res = 'map[string][]TextEdit';
      } else if (nx.getText().startsWith('[id: ChangeAnnotationIdentifier]')) {
        res = 'map[string]ChangeAnnotationIdentifier';
      } else if (nx.getText().startsWith('[uri: string')) {
        res = 'map[string]interface{}';
      } else if (nx.getText().startsWith('[uri: DocumentUri')) {
        res = 'map[DocumentURI][]TextEdit';
      } else if (nx.getText().startsWith('[key: string')) {
        res = 'map[string]interface{}';
      } else {
        throw new Error(`1100 handle ${nx.getText()} ${loc(nx)}`);
      }
      res += ` /*${comment}*/`;
      ans.push(res);
      return;
    } else
      throw new Error(`TypeLiteral had ${strKind(nx)}`);
  };
  n.members.forEach(g);
  // for some the generated type is wanted, for others it's not needed
  if (!nm.startsWith('workspace')) {
    if (res.startsWith('struct')) return res + '}';  // map[] is special
    return res;
  }
  // these names have to be made unique
  const genName = `${goName(nm)}${extraTypes.size}Gn`;
  extraTypes.set(genName, ans);
  return genName;
}

// print all the types and constants and extra types
function outputTypes() {
  // generate go types alphabeticaly
  let v = Array.from(seenTypes.keys());
  v.sort();
  v.forEach((x) => toGo(seenTypes.get(x), x));
  u.prgo(u.computeHeader(true));
  u.prgo('import "encoding/json"\n\n');
  typesOut.forEach((s) => {
    u.prgo(s);
    // it's more convenient not to have to think about trailing newlines
    // when generating types, but doc comments can't have an extra \n
    if (s.indexOf('/**') < 0) u.prgo('\n');
  });
  u.prgo('\nconst (\n');
  constsOut.forEach((s) => {
    u.prgo(s);
    u.prgo('\n');
  });
  u.prgo(')\n');
  u.prgo('// Types created to name formal parameters and embedded structs\n');
  extraTypes.forEach((v, k) => {
    u.prgo(` type ${k} struct {\n`);
    v.forEach((s) => {
      u.prgo(s);
      u.prgo('\n');
    });
    u.prgo('}\n');
  });
}

// client and server ------------------

interface side {
  methods: string[];
  cases: string[];
  calls: string[];
  name: string;    // client or server
  goName: string;  // Client or Server
  outputFile?: string;
  fd?: number
}
let client: side = {
  methods: [],
  cases: [],
  calls: [],
  name: 'client',
  goName: 'Client',
};
let server: side = {
  methods: [],
  cases: [],
  calls: [],
  name: 'server',
  goName: 'Server',
};

// commonly used output
const notNil = `if len(r.Params()) > 0 {
  return true, reply(ctx, nil, errors.Errorf("%w: expected no params", jsonrpc2.ErrInvalidParams))
}`;

// Go code for notifications. Side is client or server, m is the request
// method
function goNot(side: side, m: string) {
  if (m == '$/cancelRequest') return;  // handled specially in protocol.go
  const n = not.get(m);
  const a = goType(n.typeArguments[0], m);
  const nm = methodName(m);
  side.methods.push(sig(nm, a, ''));
  const caseHdr = ` case "${m}":  // notif`;
  let case1 = notNil;
  if (a != '' && a != 'void') {
    case1 = `var params ${a}
    if err := json.Unmarshal(r.Params(), &params); err != nil {
      return true, sendParseError(ctx, reply, err)
    }
    err:= ${side.name}.${nm}(ctx, &params)
    return true, reply(ctx, nil, err)`;
  } else {
    case1 = `err := ${side.name}.${nm}(ctx)
    return true, reply(ctx, nil, err)`;
  }
  side.cases.push(`${caseHdr}\n${case1}`);

  const arg3 = a == '' || a == 'void' ? 'nil' : 'params';
  side.calls.push(`
  func (s *${side.name}Dispatcher) ${sig(nm, a, '', true)} {
    return s.sender.Notify(ctx, "${m}", ${arg3})
  }`);
}

// Go code for requests.
function goReq(side: side, m: string) {
  const n = req.get(m);
  const nm = methodName(m);
  let a = goType(n.typeArguments[0], m);
  let b = goType(n.typeArguments[1], m);
  if (n.getText().includes('Type0')) {
    b = a;
    a = '';  // workspace/workspaceFolders and shutdown
  }
  u.prb(`${side.name} req ${a != ''}, ${b != ''} ${nm} ${m} ${loc(n)} `);
  side.methods.push(sig(nm, a, b));

  const caseHdr = `case "${m}": // req`;
  let case1 = notNil;
  if (a != '') {
    if (extraTypes.has('Param' + nm)) a = 'Param' + nm;
    case1 = `var params ${a}
    if err := json.Unmarshal(r.Params(), &params); err != nil {
      return true, sendParseError(ctx, reply, err)
    }`;
    if (a === 'ParamInitialize') {
      case1 = `var params ${a}
    if err := json.Unmarshal(r.Params(), &params); err != nil {
      if _, ok := err.(*json.UnmarshalTypeError); !ok {
        return true, sendParseError(ctx, reply, err)
      }
    }`;
    }
  }
  const arg2 = a == '' ? '' : ', &params';
  // if case2 is not explicitly typed string, typescript makes it a union of strings
  let case2: string = `if err := ${side.name}.${nm}(ctx${arg2}); err != nil {
    event.Error(ctx, "", err)
  }`;
  if (b != '' && b != 'void') {
    case2 = `resp, err := ${side.name}.${nm}(ctx${arg2})
    return true, reply(ctx, resp, err)`;
  } else {  // response is nil
    case2 = `err := ${side.name}.${nm}(ctx${arg2})
    return true, reply(ctx, nil, err)`;
  }

  side.cases.push(`${caseHdr}\n${case1}\n${case2}`);

  const callHdr = `func (s *${side.name}Dispatcher) ${sig(nm, a, b, true)} {`;
  let callBody = `return s.sender.Call(ctx, "${m}", nil, nil)\n}`;
  if (b != '' && b != 'void') {
    const p2 = a == '' ? 'nil' : 'params';
    const returnType = indirect(b) ? `*${b}` : b;
    callBody = `var result ${returnType}
			if err := s.sender.Call(ctx, "${m}", ${p2}, &result); err != nil {
				return nil, err
      }
      return result, nil
    }`;
  } else if (a != '') {
    callBody = `return s.sender.Call(ctx, "${m}", params, nil) // Call, not Notify
  }`;
  }
  side.calls.push(`${callHdr}\n${callBody}\n`);
}

// make sure method names are unique
let seenNames = new Set<string>();
function methodName(m: string): string {
  let i = m.indexOf('/');
  let s = m.substring(i + 1);
  let x = s[0].toUpperCase() + s.substring(1);
  for (let j = x.indexOf('/'); j >= 0; j = x.indexOf('/')) {
    let suffix = x.substring(j + 1);
    suffix = suffix[0].toUpperCase() + suffix.substring(1);
    let prefix = x.substring(0, j);
    x = prefix + suffix;
  }
  if (seenNames.has(x)) {
    // various Resolve and Diagnostic
    x += m[0].toUpperCase() + m.substring(1, i);
  }
  seenNames.add(x);
  return x;
}

// used in sig and in goReq
function indirect(s: string): boolean {
  if (s == '' || s == 'void') return false;
  const skip = (x: string) => s.startsWith(x);
  if (skip('[]') || skip('interface') || skip('Declaration') ||
    skip('Definition') || skip('DocumentSelector'))
    return false;
  return true;
}

// Go signatures for methods.
function sig(nm: string, a: string, b: string, names?: boolean): string {
  if (a.indexOf('struct') != -1) {
    const v = a.split('\n');
    extraTypes.set(`Param${nm}`, v.slice(1, v.length - 1));
    a = 'Param' + nm;
  }
  if (a == 'void')
    a = '';
  else if (a != '') {
    if (names)
      a = ', params *' + a;
    else
      a = ', *' + a;
  }
  let ret = 'error';
  if (b != '' && b != 'void') {
    // avoid * when it is senseless
    if (indirect(b)) b = '*' + b;
    ret = `(${b}, error)`;
  }
  let start = `${nm}(`;
  if (names) {
    start = start + 'ctx ';
  }
  return `${start}context.Context${a}) ${ret}`;
}

// write the request/notification code
function output(side: side) {
  // make sure the output file exists
  if (!side.outputFile) {
    side.outputFile = `ts${side.name}.go`;
    side.fd = fs.openSync(side.outputFile, 'w');
  }
  const f = function (s: string) {
    fs.writeSync(side.fd!, s);
    fs.writeSync(side.fd!, '\n');
  };
  f(u.computeHeader(false));
  f(`
        import (
          "context"
          "encoding/json"

          "golang.org/x/tools/internal/jsonrpc2"
          errors "golang.org/x/xerrors"
        )
        `);
  const a = side.name[0].toUpperCase() + side.name.substring(1);
  f(`type ${a} interface {`);
  side.methods.forEach((v) => { f(v); });
  f('}\n');
  f(`func ${side.name}Dispatch(ctx context.Context, ${side.name} ${a}, reply jsonrpc2.Replier, r jsonrpc2.Request) (bool, error) {
          switch r.Method() {`);
  side.cases.forEach((v) => { f(v); });
  f(`
        default:
          return false, nil
        }
      }`);
  side.calls.forEach((v) => { f(v); });
}

// Handling of non-standard requests, so we can add gopls-specific calls.
function nonstandardRequests() {
  server.methods.push(
    'NonstandardRequest(ctx context.Context, method string, params interface{}) (interface{}, error)');
  server.calls.push(
    `func (s *serverDispatcher) NonstandardRequest(ctx context.Context, method string, params interface{}) (interface{}, error) {
      var result interface{}
      if err := s.sender.Call(ctx, method, params, &result); err != nil {
        return nil, err
      }
      return result, nil
    }
  `);
}

// ----- remember it's a scripting language
function main() {
  if (u.gitHash != u.git()) {
    throw new Error(
      `git hash mismatch, wanted\n${u.gitHash} but source is at\n${u.git()}`);
  }
  u.createOutputFiles();
  parse();
  u.printAST(program);
  // find the Requests and Nofificatations
  for (const sourceFile of program.getSourceFiles()) {
    if (!sourceFile.isDeclarationFile) {
      ts.forEachChild(sourceFile, findRPCs);
    }
  }
  // separate RPCs into client and server
  setReceives();
  // visit every sourceFile collecting top-level type definitions
  for (const sourceFile of program.getSourceFiles()) {
    if (!sourceFile.isDeclarationFile) {
      ts.forEachChild(sourceFile, genTypes);
    }
  }
  // check that each thing occurs exactly once, and put pointers into
  // seenTypes
  checkOnce();
  // for each of Client and Server there are 3 parts to the output:
  // 1. type X interface {methods}
  // 2. func (h *serverHandler) Deliver(...) { switch r.method }
  // 3. func (x *xDispatcher) Method(ctx, parm)
  not.forEach(  // notifications
    (v, k) => {
      receives.get(k) == 'client' ? goNot(client, k) : goNot(server, k);
    });
  req.forEach(  // requests
    (v, k) => {
      receives.get(k) == 'client' ? goReq(client, k) : goReq(server, k);
    });
  nonstandardRequests();
  // find all the types implied by seenTypes and rpcs to try to avoid
  // generating types that aren't used
  moreTypes();
  // do merging
  cleanData();
  // and print the Go code
  outputTypes();
  console.log(`seen ${seenTypes.size + extraTypes.size}`);
  output(client);
  output(server);
}

main();
