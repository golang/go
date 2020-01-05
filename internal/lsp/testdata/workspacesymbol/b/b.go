package b

var WorkspaceSymbolVariableB = "b" //@symbol("WorkspaceSymbolVariableB", "WorkspaceSymbolVariableB", "Variable", "")

type WorkspaceSymbolStructB struct { //@symbol("WorkspaceSymbolStructB", "WorkspaceSymbolStructB", "Struct", "")
	Bar int //@mark(bBar, "Bar"), symbol("Bar", "Bar", "Field", "WorkspaceSymbolStructB")
}
