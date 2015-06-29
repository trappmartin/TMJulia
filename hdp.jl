using Distributions

function parseCorpus(dpath, cpath)
	  dfile = open(dpath, "r")
	  cfile = open(cpath, "r")

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

	  dictionary = map(chomp, readlines(dfile))
	  dictionary, corpus
end

function sampletopicindex(ndk, nd, nkw, nk, d, word, W, K, tau, alpha, beta)
	  if K == 0
		    return 1
	  end
	  p = zeros(K+1)
	  p[1] = (ndk[d, 1] + tau[1] * alpha) * (nkw[1, word] + beta)/(nk[1] + W * beta)

	  for k in 2:K
		    p[k] = p[k-1] + (ndk[d, k] + tau[k] * alpha) * (nkw[k, word] + beta)/(nk[k] + W * beta)
	  end

	  p[K+1] = p[K] + (tau[K+1] * alpha / W)

	  u = p[K+1]*rand()
	  for k in 1:K+1
		    if u < p[k]
            return k
		    end
	  end
end

function initialize(dictionary, corpus, W, Kmax, alpha, beta, gamma)
	  K = 0
	  topics = deepcopy(corpus)
	  ndk = zeros(Int64, length(topics), K)
	  nd = zeros(Int64, length(topics))
	  nkw = zeros(Int64, K, length(dictionary))
	  nk = zeros(Int64, K)
	  U1 = Int64[]
	  U0 = [1:Kmax]
	  tau = 1
	  for (d, doc) in enumerate(corpus)
		    for (w, word) in enumerate(doc)
			      k = sampletopicindex(ndk, nd, nkw, nk, d, word, W, K, tau, alpha, beta)


			      if k > K
				        try
					          _k = shift!(U0)
				        catch
					          U0 = [Kmax+1:Kmax*2]
					          Kmax *= 2
					          _k = shift!(U0)
				        end
                topics[d][w] = _k
				        push!(U1, _k)
				        if _k == K + 1

					          _ndk = zeros(Int64, length(corpus))
					          _ndk[d] += 1
					          ndk = [ndk _ndk]
					          nd[d] += 1

					          _nkw = zeros(Int64, W)
					          _nkw[word] += 1
					          nkw = [nkw; _nkw']

					          push!(nk, 1)
					          tau = rand(Dirichlet(vec([nk .+ beta, gamma])), 1)
				        else
					          ndk[d, _k] += 1
					          nd[d] += 1
					          nkw[_k, word] += 1
					          nk[_k] += 1
				        end
				        K += 1
			      else
                _k = U1[k]
                topics[d][w] = _k
				        ndk[d, _k] += 1
				        nd[d] += 1
				        nkw[_k, word] += 1
				        nk[_k] += 1
			      end
		    end
	  end


	  topics, ndk, nd, nkw, nk, K, Kmax, U1, U0, tau
end

function gibbs(corpus, topics, ndk, nd, nkw, nk, W, K, Kmax, U1, U0, tau, alpha, beta, gamma, iterations)
	KOverTime = Int64[]
	  for n in 1:iterations
		push!(KOverTime, K)
		    println("Iteration $n")
			println(length(KOverTime))
        println(nk)
        println(K)
		    for (d, doc) in enumerate(corpus)
			      for (w, word) in enumerate(doc)
				        k = topics[d][w]
				        ndk[d, k] -= 1
				        nd[d] -= 1
				        nkw[k, word] -= 1
				        nk[k] -= 1

				        k = sampletopicindex(ndk, nd, nkw, nk, d, word, W, K, tau, alpha, beta)

				        if k > K
					          try
						            _k = shift!(U0)
					          catch
						            U0 = [Kmax+1:Kmax*2]
						            Kmax *= 2
						            _k = shift!(U0)
					          end
                    topics[d][w] = _k
					          push!(U1, _k)
					          if _k == K + 1
						            _ndk = zeros(Int64, length(corpus))
						            _ndk[d] += 1
						            ndk = [ndk _ndk]
						            nd[d] += 1

						            _nkw = zeros(Int64, W)
						            _nkw[word] += 1
						            nkw = [nkw; _nkw']

						            push!(nk, 1)

						            tau = rand(Dirichlet(vec([nk .+ beta, gamma])), 1)
					          else
						            ndk[d, _k] += 1
						            nd[d] += 1
						            nkw[_k, word] += 1
						            nk[_k] += 1
					          end
					          K += 1
				        else
					          _k = U1[k]
                    topics[d][w] = _k
					          ndk[d, _k] += 1
					          nd[d] += 1
					          nkw[_k, word] += 1
					          nk[_k] += 1
				        end
			      end
		    end

        # offset to account for variable stack size U1 
        offset = 0
		    for k in 1:K
			      if nk[k - offset] == 0
				        deleteat!(U1, k - offset)
                U1[k - offset:end] -= 1
				        unshift!(U0, K)
                ndk = [ndk[:, 1:(k - offset - 1)] ndk[:, (k - offset + 1):end]]
                nkw = [nkw[1:(k - offset - 1), :]; nkw[(k - offset + 1):end, :]]
                deleteat!(nk, k - offset)
				        K -= 1
                offset += 1

                for i = 1:length(topics)
                    for j = 1:length(topics[i])
                        if topics[i][j] > k - offset
                            topics[i][j] -= 1
                        end
                    end
                end
			      end
		    end 
		    tau = rand(Dirichlet(vec([nk .+ beta, gamma])), 1)
    end
    topics, ndk, nd, nkw, nk, K, Kmax, U1, U0, tau, KOverTime
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

function perplexity(dictionary, corpus, phi, theta)
    logprob_c = 0
    for (d, document) in enumerate(corpus)
        logprob_d = 0
        nw = hist(document,0.5:maximum(document)+0.5)[2]
        for word in unique(document)
            logprob_d += nw[word] * log(sum(vec(phi[:,word]) .* vec(theta[d,:])))
        end
        logprob_c += logprob_d
    end
    println(logprob_c)
    logprob_c /= sum(map(length, corpus))
    println(logprob_c)
    exp(-1 * logprob_c)
end

function printTopics(phi, dictionary, nwords, K)
    for k = 1:K
        topic = hcat(phi'[:,k], dictionary)
        println("Topic $k:\n $(sortrows(topic, rev=true, by=x->(x[1]))[1:nwords,:])")
    end
end

using HDF5, JLD

(dictionary, corpus) = parseCorpus("data/reuters_vocab.dat", "data/reuters_train.dat")
for gamma=0:5:100
    println("Running HDP for gamma=", gamma)
    (topics, ndk, nd, nkw, nk, K, Kmax, U1, U0, tau) = initialize(dictionary, corpus, length(dictionary), 1000, 1, 0.1, max(gamma, 1))
    (topics, ndk, nd, nkw, nk, K, Kmax, U1, U0, tau, KOverTime) = gibbs(corpus, topics, ndk, nd, nkw, nk, length(dictionary), K, Kmax, U1, U0, tau, 1, 0.1, max(gamma, 1), 1000)
    save(string("models/hdp/hdp_1_01_",gamma), "Z", topics, "ndk", ndk, "nd", nd, "nkw", nkw, "nk", nk, "k_over_time", KOverTime)
end


# using ArgParse
# s = ArgParseSettings()
# @add_arg_table s begin
#     "--alpha", "-a"
#     help = "The alpha hyperparameter"
#     arg_type = Number
#     default = 1
#     "--beta", "-b"
#     help = "The beta hyperparameter"
#     arg_type = Number
#     default = 0.01
#     "--gamma", "-g"
#     help = "The gamma hyperparameter"
#     arg_type = Number
#     default = 1
#     "--iterations", "-i"
#     help = "The number of iterations"
#     arg_type = Int
#     required = true
#     "--dictionary", "-d"
#     help = "Path to vocabulary file"
#     "corpus"
#     help = "Path to corpus file"
#     required = true
# end

# parsed_args = parse_args(s)
# println("Running HDP with the following settings:\n$parsed_args")
#                                                                    (dictionary, corpus) = parseCorpus(parsed_args["dictionary"], parsed_args["corpus"])

# (topics, ndk, nd, nkw, nk, K, Kmax, U1, U0, tau) = initialize(dictionary, corpus, length(dictionary), 1000, parsed_args["alpha"], parsed_args["beta"], parsed_args["gamma"])

# (topics, ndk, nd, nkw, nk, K, Kmax, U1, U0, tau) = gibbs(corpus, topics, ndk, nd, nkw, nk, length(dictionary), K, Kmax, U1, U0, tau, parsed_args["alpha"], parsed_args["beta"], parsed_args["gamma"], parsed_args["iterations"])

# printTopics(estimatePhi(nkw, nk, parsed_args["beta"], dictionary), dictionary, 10, K)

# println(perplexity(dictionary, corpus, estimatePhi(nkw, nk, parsed_args["beta"], dictionary), estimateTheta(ndk, parsed_args["alpha"], K)))
