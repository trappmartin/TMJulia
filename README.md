# TMJulia

Just run the script, for example:

~~~
julia lda.jl -k 5 -a 1 -b 0.1 -i 10 -d "data/vocab.txt" "data/corpus.dat"
~~~

positional arguments:

name | help
-----| ----
corpus | Path to corpus file

named arguments:

name | help
---- | ----
-k, --num_topics |  The number of topics (type: Int64)
-a, --alpha | The alpha hyperparameter (type: Number, default: 1)
-b, --beta | The beta hyperparameter (type: Number, default: 0.01)
-i, --iterations | The number of iterations (type: Int64)
-d, --dictionary | Path to vocabulary file
-h, --help | Show this help message and exit


Or for the biterm topic model:

~~~
julia btm.jl
~~~

No command-line arguments yet.
