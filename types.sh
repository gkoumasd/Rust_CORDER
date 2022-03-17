grep type ~/.tree-sitter/bin/tree-sitter-asm-0.0.1/src/node-types.json | awk '{split($2, a, /"/); print a[2]}' | sort | uniq > vocab/treesitter_asm/node_type/type.txt

