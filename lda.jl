function parseCorpus(dpath, cpath, tpath)
    dfile = open(dpath, "r")
    cfile = open(cpath, "r")

    train = Array(Vector{Int64}, countlines(cpath))
    for (i, line) in enumerate(eachline(cfile))
        line = split(line)
        document = Int64[]
        for wordcount in line
            (word, count) = split(wordcount, ":")
            for _ in 1:parseint(count)
                push!(document, parseint(word) + 1)
            end
        end
        train[i] = document
    end

    if tpath != false 

        tfile = open(tpath, "r")
        test = Array(Vector{Int64}, countlines(tpath))

        for (i, line) in enumerate(eachline(tfile))
            line = split(line)
            document = Int64[]
            for wordcount in line
                (word, count) = split(wordcount, ":")
                for _ in 1:parseint(count)
                    push!(document, parseint(word) + 1)
                end
            end
            test[i] = document
        end
    end
    dictionary = map(chomp, readlines(dfile))
    dictionary, train, test
end

function initialize(dictionary, corpus, K)
    topics = deepcopy(corpus)
    ndk = zeros(Int64, length(topics), K)
    nkw = zeros(Int64, K, length(dictionary))
    nk = zeros(Int64, K)
    for doc in 1:length(topics)
        for wordpos in 1:length(topics[doc])
            word = corpus[doc][wordpos]
            topic = rand(1:K)
            topics[doc][wordpos] = topic
            ndk[doc, topic] += 1
            nkw[topic, word] += 1
            nk[topic] += 1
        end
    end
    topics, ndk, nkw, nk
end

function sampleTopic(ndk, nkw, nk, document, word, K, W, alpha, beta)
    p = zeros(K)
    p[1] = (ndk[document, 1] + alpha) * ((nkw[1, word] + beta)/(nk[1] + (W * beta)))
    for k = 2:K 
        p[k] = p[k-1] + (ndk[document, k] + alpha) * ((nkw[k, word] + beta)/(nk[k] + (W * beta)))
    end
    u = p[K]*rand()
    for topic in 1:K
        if u < p[topic]
            return topic
        end
    end
end

function gibbs(dictionary, corpus, topics, ndk, nkw, nk, K, alpha, beta, iterations)
    W = length(dictionary)
    for n = 1:iterations
        println("Iteration $n")
        for (i, doc) in enumerate(corpus)

            for (j, word) in enumerate(doc)

                topic = topics[i][j]
                ndk[i, topic] -= 1
                nkw[topic, word] -= 1
                nk[topic] -= 1

                topic = sampleTopic(ndk, nkw, nk, i, word, K, W, alpha, beta)

                topics[i][j] = topic
                ndk[i, topic] += 1
                nkw[topic, word] += 1
                nk[topic] += 1
            end
        end
    end
    topics, ndk, nkw, nk
end

function estimateTheta(ndk, alpha, K)
    thetadk = Array(Float64, size(ndk)...)
    for d in 1:size(thetadk, 1)
        for k in 1:size(thetadk, 2)
            thetadk[d,k] = (ndk[d,k] + alpha) / (sum(ndk, 2)[d] + (K * alpha))
        end
    end
    thetadk
end

function estimatePhi(nkw, nk, beta, dictionary)
    phikw = Array(Float64, size(nkw)...)
    for k in 1:size(phikw, 1)
        for w in 1:size(phikw, 2)
            phikw[k,w] = (nkw[k,w] + beta)/(nk[k] + (length(dictionary) * beta))
        end
    end
    phikw
end

function perplexity(test, phi, theta)
    logprob_c = 0

    for (d, document) in enumerate(test)
        _theta = theta[rand(1:size(theta, 1)), :]
        logprob_d = 0
        nw = hist(document,0.5:maximum(document)+0.5)[2]
        for word in unique(document)
            logprob_d += nw[word] * log(sum(vec(phi[:,word]) .* vec(_theta)))
        end
        println(logprob_d)
        logprob_c += logprob_d
    end
    logprob_c /= sum(map(length, test))
    exp(-1 * logprob_c)
end


function printTopics(phi, dictionary, nwords, K)
    for k = 1:K
        topic = hcat(phi'[:,k], dictionary)
        println("Topic $k:\n $(sortrows(topic, rev=true, by=x->(x[1]))[1:nwords,:])")
    end
end

using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "--num_topics", "-k"
    help = "The number of topics"
    arg_type = Int
    required = true
    "--alpha", "-a"
    help = "The alpha hyperparameter"
    arg_type = Number
    default = 1
    "--beta", "-b"
    help = "The beta hyperparameter"
    arg_type = Number
    default = 0.01
    "--iterations", "-i"
    help = "The number of iterations"
    arg_type = Int
    required = true
    "--dictionary", "-d"
    help = "Path to vocabulary file"
    "--test", "-t"
    help = "Test data"
    "corpus"
    help = "Path to corpus file"
    required = true
end

parsed_args = parse_args(s)

println("Running LDA with the following settings:\n$parsed_args")

(dictionary, corpus, test) = parseCorpus(parsed_args["dictionary"], parsed_args["corpus"], parsed_args["test"])
(Z, ndk, nkw, nk) = initialize(dictionary, corpus, parsed_args["num_topics"])
gibbs(dictionary, corpus, Z, ndk, nkw, nk, parsed_args["num_topics"], parsed_args["alpha"], parsed_args["beta"], parsed_args["iterations"])

printTopics(estimatePhi(nkw, nk, parsed_args["beta"], dictionary), dictionary, 10, parsed_args["num_topics"])

println(perplexity(test, estimatePhi(nkw, nk, parsed_args["beta"], dictionary), estimateTheta(ndk, parsed_args["alpha"], parsed_args["num_topics"])))
