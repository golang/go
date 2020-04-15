package b

var WorkspaceSymbolVariableB = "b" //@symbol("WorkspaceSymbolVariableB", "WorkspaceSymbolVariableB", "Variable", "", "WorkspaceSymbolVariableB")

type WorkspaceSymbolStructB struct { //@symbol("WorkspaceSymbolStructB", "WorkspaceSymbolStructB", "Struct", "", "WorkspaceSymbolStructB")
	Bar int //@mark(bBar, "Bar"), symbol("Bar", "Bar", "Field", "WorkspaceSymbolStructB", "Bar")
}
