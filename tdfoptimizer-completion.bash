# Autocompletion voor tdfoptimizer.py
_tdfoptimizer_complete() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    opts="--sep --race --max-price --csv-out --cache --test"

    if [[ ${cur} == -* ]]; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
}
complete -F _tdfoptimizer_complete python
