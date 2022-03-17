<<<<<<< HEAD
grep type ~/.tree-sitter/bin/tree-sitter-asm-0.0.1/src/node-types.json | awk '{split($2, a, /"/); print a[2]}' | sort | uniq > vocab/treesitter_asm/node_type/type.txt

=======
~/Documents/github.com/yijunyu/tree-sitter-parsers/tree-sitter-asm-0.0.1/types.sh
mv types.txt vocab/treesitter_asm/node_type/type.txt
>>>>>>> 2ea611c0e846c1162439d46c62e9de4525718476
