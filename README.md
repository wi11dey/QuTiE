# QuTiE

## Installation

1. Install [Julia](https://julialang.org/).
1. In the project root directory, run the following to install all dependencies
```
$ julia --project=.
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.8.3 (2022-11-14)
 _/ |\__'_|_|_|\__'_|  |  
|__/                   |

julia>]instantiate
```

## Usage

```shell
$ ./QuTiE specification.jl
```

QuTiE will watch the specification file (written in the QuTiE language) provided on the command line and automatically update the numerical simulation accordingly. Please see the thesis document for a description of QuTiE syntax and examples of usage.
