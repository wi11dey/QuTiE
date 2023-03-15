using MacroTools

export @define

"""Attaches name information to the given object."""
define(name::Symbol, x) = x
macro define(expr)
    @capture(expr, name_Symbol := definition_) || error("Invalid definition")
    definition = MacroTools.postwalk(definition) do fragment
        Meta.isexpr(fragment, :parameters) || return fragment
        Expr(:parameters, (kw isa Symbol ? Expr(:kw, kw, true) : kw for kw in fragment.args)...)
    end
    :(const $(esc(name)) = $define($(Meta.quot(name)), $(esc(definition))))
end
