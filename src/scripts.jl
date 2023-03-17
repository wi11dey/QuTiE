import REPL.REPLCompletions: superscripts, subscripts

sup(s::AbstractString) = replace(s, superscripts...)
sub(s::AbstractString) = replace(s, subscripts...)
sup(n::Integer) = n |> string |> sup
sub(n::Integer) = n |> string |> sub
