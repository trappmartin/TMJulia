function parseCorpus(dpath, cpath, tpath)
    dfile = open(dpath, "r")
    cfile = open(cpath, "r")
    tfile = false
    test = false
    if tpath != false 
        tfile = open(tpath, "r")
        test = Array(Vector{Int64}, countlines(tpath))
    end
    corpus = Array(Vector{Int64}, countlines(cpath))
    for (i, line) in enumerate(eachline(cfile))
        line = split(line)
        document = Int64[]
        for wordcount in line
            (word, count) = split(wordcount, ":")
            for _ in 1:parseint(count)
                push!(document, parseint(word) + 1)
            end
        end
        corpus[i] = document
    end

    if tpath != false 
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
    dictionary, corpus, test
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

function gibbs(dictionary, corpus, test, topics, ndk, nkw, nk, K, alpha, beta, iterations)
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

function perplexity(dictionary, corpus, test, phi, theta)
    logprob_c = 0

    for (d, document) in enumerate(test)
        _theta = theta[rand(1:size(theta, 1)), :]
        logprob_d = 0
        nw = hist(document,0.5:maximum(document)+0.5)[2]
        for word in unique(document)
            logprob_d += nw[word] * log(sum(vec(phi[:,word]) .* vec(_theta)))
        end
        logprob_c += logprob_d
    end
    logprob_c /= sum(map(length, test))
    exp(-1 * logprob_c)
end

function log_Tprob_base(zto, zfrom, Nk, doc, phi, alpha)
    lp = 0
    for (i, word) in enumerate(doc)
        Nk[zfrom[i]] -= 1
        P = phi[:, word] .* (Nk + alpha)
        P /= sum(P)
        lp += log(P[zto[i]])
        Nk[zto[i]] += 1
    end
    lp
end

function chib_estimate(doc, phi, alpha, K, V, iters=1000, burn_iters=3)
    Nk = zeros(Int64, K)
    P = zeros(Number, K)
    topics = zeros(Int64, length(doc))
    for (i, word) in enumerate(doc)
        P = phi[:, word] * alpha
        u = P[end]*rand()
        for topic in 1:K
            if u < P[topic]
                topics[i] = topic
                Nk[topic] += 1
                break
            end
        end
    end
    # Run some sweeps of Gibbs sampling
    for sweeps = 1:burn_iters
        for (i, word) in enumerate(doc)
            Nk[topics[i]] -= 1
            P = phi[:, word] .* (Nk + alpha)
            u = P[end]*rand()
            for topic in 1:K
                if u < P[topic]
                    topics[i] = topic
                    Nk[topic] += 1
                    break
                end
            end
        end
    end

    for i = 1:12 # Magic number taken from original Matlab code
        old_topics = copy(topics)
        for (i, word) in enumerate(doc)
            Nk[topics[i]] -= 1
            P = phi[:, word] .* (Nk + alpha)
            topics[i] = indmax(P)
            Nk[topics[i]] += 1
        end
        if old_topics == topics
            break
        end
    end

    # run Murray & Salakhutdinov algorithm
    topics_star = copy(topics)
    log_Tvals = zeros(Number, iters)
    log_Tprob(zto, zfrom, Nzfrom) = log_Tprob_base(zto, zfrom, Nzfrom, doc, phi, alpha)

    # draw starting position
    ss = ceil(rand() * iters)
   # Draw Z^(s)
for i in length(doc):-1:1
    Nk[topics[i]] -= 1
    P = phi[:, doc[i]] .* (Nk + alpha)
    topics[i] = indmax(P)
    Nk[topics[i]] += 1
end
# end
topics_s = copy(topics)
Ns = copy(Nk)
log_Tvals[ss] = log_Tprob(topics_star, topics, Nk)

# Draw forward stuff
for sprime = (ss+1):iters
    for (i, word) in enumerate(doc)
        Nk[topics[i]] -= 1
        P = phi[:, word] .* (Nk + alpha)
        topics[i] = indmax(P)
        Nk[topics[i]] += 1
    end
    log_Tvals[sprime] = log_Tprob(topics_star, topics, Nk)
end

# Go back to middle

topics = copy(topics_s)
Nk = copy(Ns)

for sprime = (ss-1):-1:1
    for (i, word) in enumerate(doc)
        Nk[topics[i]] -= 1
        P = phi[:, word] .* (Nk + alpha)
        topics[i] = indmax(P)
        Nk[topics[i]] += 1
    end
    log_Tvals[sprime] = log_Tprob(topics_star, topics, Nk)
end

# final estimate
Nkstar = hist(topics_star,0.5:K+0.5)[2]
log_pz = sum(log(gamma(Nkstar + alpha))) + log(gamma(alpha)) - sum(log(gamma(alpha))) - log(gamma(length(doc)))
log_w_given_z = 0

for i = 1:length(doc)
    log_w_given_z += log(phi[topics_star[i], doc[i]])
end
log_joint = log_pz + log_w_given_z
log_joint - (log(sum(exp(log_Tvals))) - log(iters))

end

function printTopics(phi, dictionary, nwords, K)
    for k = 1:K
        topic = hcat(phi'[:,k], dictionary)
        println("Topic $k:\n $(sortrows(topic, rev=true, by=x->(x[1]))[1:nwords,:])")
    end
end

using HDF5, JLD

(dictionary, corpus, test) = parseCorpus("data/feelings_vocab.dat", "data/feelings.dat", "data/feelings.dat")

# for K = 10:10:200
#     println("Running LDA for ", K, " topics")
#     (Z, ndk, nkw, nk) = initialize(dictionary, corpus, K)
#     (Z, ndk, nkw, nk) = gibbs(dictionary, corpus, test, Z, ndk, nkw, nk, K, 1, 0.1, 1000)
#     save(string("models_feelings/lda/lda_1_01_", K), "Z", Z, "ndk", ndk, "nkw", nkw, "nk", nk)
# end

# using ArgParse

# s = ArgParseSettings()
# @add_arg_table s begin
#     "--num_topics", "-k"
#     help = "The number of topics"
#     arg_type = Int
#     required = true
#     "--alpha", "-a"
#     help = "The alpha hyperparameter"
#     arg_type = Number
#     default = 1
#     "--beta", "-b"
#     help = "The beta hyperparameter"
#     arg_type = Number
#     default = 0.01
#     "--iterations", "-i"
#     help = "The number of iterations"
#     arg_type = Int
#     required = true
#     "--dictionary", "-d"
#     help = "Path to vocabulary file"
#     "--test", "-t"
#     help = "Test data"
#     "corpus"
#     help = "Path to corpus file"
#     required = true
# end

# parsed_args = parse_args(s)

# println("Running LDA with the following settings:\n$parsed_args")

# (dictionary, corpus, test) = parseCorpus(parsed_args["dictionary"], parsed_args["corpus"], parsed_args["test"])
# (Z, ndk, nkw, nk) = initialize(dictionary, corpus, parsed_args["num_topics"])
# gibbs(dictionary, corpus, test, Z, ndk, nkw, nk, parsed_args["num_topics"], parsed_args["alpha"], parsed_args["beta"], parsed_args["iterations"])

# printTopics(estimatePhi(nkw, nk, parsed_args["beta"], dictionary), dictionary, 10, parsed_args["num_topics"])

# if test != false
#     println(perplexity(dictionary, test, estimatePhi(nkw, nk, parsed_args["beta"], dictionary), estimateTheta(ndk, parsed_args["alpha"], parsed_args["num_topics"])))
# else
#     println(perplexity(dictionary, corpus, estimatePhi(nkw, nk, parsed_args["beta"], dictionary), estimateTheta(ndk, parsed_args["alpha"], parsed_args["num_topics"])))
# end
