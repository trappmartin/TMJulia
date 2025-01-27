using Distributions

function parseCorpus(dpath, cpath, window)
	  dfile = open(dpath, "r")
	  cfile = open(cpath, "r")

	  dictionary = map(chomp, readlines(dfile))

	  biterms = Array((Int64, Int64), 0)
	  for (i, line) in enumerate(eachline(cfile))
		    line = split(line)
		    document = Int64[]
		    for wordcount in line
			      (word, count) = split(wordcount, ":")
			      for _ in 1:parseint(count)
				        push!(document, parseint(word) + 1)
			      end
		    end
		    for i in 1:window:length(document)-1
			      for j in i:min(i+(window-2),length(document)-1)
				        for k in j+1:min(i+(window-1), length(document))
					          push!(biterms, (document[j], document[k]))
				        end
			      end
		    end
	  end
	  dictionary, biterms
end

function sampletopicindex(nk, nwz, w1, w2, W, K, tau, alpha, beta, delta)
    if K == 0
        return 1
    end
    p = zeros(K+1)

    p[1] = (nk[1] + tau[1] * alpha) * (max(nwz[w1, 1] - delta, 0) + beta) * (max(nwz[w2, 1] - delta, 0) + beta) / ((2 * nk[1] + W * beta) * (2 * nk[1] + (W + 1) * beta))
	for k = 2:K 
		p[k] = p[k-1] + (nk[k] + tau[k] * alpha) * (max(nwz[w1, k] - k*delta, 0) + beta) * (max(nwz[w2, k] - delta, 0) + beta) / ((2 * nk[k] + W * beta) * (2 * nk[k] + (W + 1) * beta))
	end

    p[K+1] = p[K] + (tau[K+1] * (alpha + K*delta) / W)

	  u = p[K+1]*rand()
	  for k in 1:K+1
		    if u < p[k]
            return k
		    end
    end
end

function initialize(dictionary, biterms, W, Kmax, alpha, beta, gamma, delta)
    K = 0
	  topics = Array(Int64, length(biterms))
	  nk = zeros(Int64, K)
    nwz = zeros(Int64, length(dictionary), K)
    U1 = Int64[]
    U0 = [1:Kmax]
    tau = 1
    for (i, biterm) in enumerate(biterms)
        k = sampletopicindex(nk, nwz, biterm[1], biterm[2], W, K, tau, alpha, beta, delta)
        if k > K
            try
                _k = shift!(U0)
            catch
                U0 = [Kmax+1:Kmax*2]
                Kmax *= 2
                _k = shift!(U0)
            end
            topics[i] = _k
            push!(U1, _k)
            if _k == K + 1

                push!(nk, 1)

                _nwz = zeros(Int64, W)
                _nwz[biterm[1]] += 1
                _nwz[biterm[2]] += 1
                nwz = [nwz _nwz]
                tau = rand(Dirichlet(vec([nk .+ beta, gamma])), 1)
            else
                nk[_k] += 1
                nkw[biterm[1], _k] += 1
                nkw[biterm[2], _k] += 1
            end
            K += 1
        else
            _k = U1[k]
            topics[i] = _k
            nk[_k] += 1
            nwz[biterm[1], _k] += 1
            nwz[biterm[2], _k] += 1
        end
    end
	  topics, nk, nwz, K, Kmax, U1, U0, tau
end

function gibbs(corpus, topics, nk, nwz, W, K, Kmax, U1, U0, tau, alpha, beta, gamma, delta, iterations)
	KOverTime = Int64[]
    for n in 1:iterations
			push!(KOverTime, K)
        println("Iteration $n")
        for (i, biterm) in enumerate(biterms)
            w1 = biterm[1]
            w2 = biterm[2]
            k = topics[i]
            nk[k] -= 1
            nwz[w1, k] -= 1
            nwz[w2, k] -= 1
            k = sampletopicindex(nk, nwz, w1, w2, W, K, tau, alpha, beta, delta)
            if k > K
                try
                    _k = shift!(U0)
                catch
                    U0 = [K+1:K*2]
                    _k = shift!(U0)
                end
                topics[i] = _k
                push!(U1, _k)
                if _k == K + 1
                    push!(nk, 1)

                    _nwz = zeros(Int64, W)
                    _nwz[biterm[1]] += 1
                    _nwz[biterm[2]] += 1

                    nwz = [nwz _nwz]

                    tau = rand(Dirichlet(vec([nk .+ beta, gamma])), 1)
                else
                    nk[_k] += 1
                    nkw[biterm[1], _k] += 1
                    nkw[biterm[2], _k] += 1
                end
                K += 1
            else
                _k = U1[k]
                topics[i] = _k
                nk[_k] += 1
                nwz[biterm[1], _k] += 1
                nwz[biterm[2], _k] += 1
            end
        end
        # offset to account for variable stack size U1 
        offset = 0
        for k in 1:K
            if nk[k - offset] == 0
                deleteat!(U1, k - offset)
				U1[k - offset:end] -= 1
				unshift!(U0, k)
				nwz = [nwz[:, 1:(k - offset - 1)] nwz[:, (k - offset + 1):end]]
				deleteat!(nk, k - offset)
				K -= 1
				offset += 1
				for i = 1:length(topics)
                        if topics[i] > k - offset
                            topics[i] -= 1
                    end
                end
			end
		end
			U0 = [K+1:K*2]
		    tau = rand(Dirichlet(vec([nk .+ beta, gamma])), 1)
    end
    topics, nk, nwz, K, Kmax, U1, U0, tau, KOverTime
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


function printTopics(phi, dictionary, nwords, K)
    for k = 1:K
        topic = hcat(phi'[:,k], dictionary)
        println("Topic $k:\n $(sortrows(topic, rev=true, by=x->(x[1]))[1:nwords,:])")
    end
end

using ArgParse
s = ArgParseSettings()
@add_arg_table s begin
    "--alpha", "-a"
    help = "The alpha hyperparameter"
    arg_type = Number
    default = 1
    "--beta", "-b"
    help = "The beta hyperparameter"
    arg_type = Number
    default = 0.01
    "--gamma", "-g"
    help = "The gamma hyperparameter"
    arg_type = Number
    default = 1
    "--delta", "-d"
    help = "The delta hyperparameter"
    arg_type = Number
    default = 0.001
    "--iterations", "-i"
    help = "The number of iterations"
    arg_type = Int
    required = true
    "--dictionary", "-v"
    help = "Path to vocabulary file"
    "corpus"
    help = "Path to corpus file"
    required = true
    "--window", "-w"
    help = "The context window in which co-occurrences are considered"
    arg_type = Int
    default = 15
end

parsed_args = parse_args(s)
println("Running BTM with the following settings:\n$parsed_args")
(dictionary, biterms) = parseCorpus(parsed_args["dictionary"], parsed_args["corpus"], parsed_args["window"])
(topics, nk, nwz, K, Kmax, U1, U0, tau) = initialize(dictionary, biterms, length(dictionary), parsed_args["iterations"], parsed_args["alpha"], parsed_args["beta"], parsed_args["gamma"], parsed_args["delta"])
(topics, nk, nwz, K, Kmax, U1, U0, tau, KOverTime) = gibbs(biterms, topics, nk, nwz, length(dictionary), K, Kmax, U1, U0, tau, parsed_args["alpha"], parsed_args["beta"], parsed_args["gamma"], parsed_args["delta"], parsed_args["iterations"])
printTopics(estimatePhi(nwz', nk, parsed_args["beta"], dictionary), dictionary, 10, K)
