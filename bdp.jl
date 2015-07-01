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
	println(length(biterms), " biterms")
	dictionary, biterms
end

function sampletopicindex(nk, nwz, w1, w2, W, K, tau, alpha, beta)
    if K == 0
        return 1
    end
    p = zeros(K+1)

    p[1] = (nk[1] + tau[1] * alpha) * ((nwz[w1, 1] + beta) * (nwz[w2, 1] + beta)) / ((2 * nk[1] + W * beta) * (2 * nk[1] + (W + 1) * beta))
	for k = 2:K 
	try
		p[k] = p[k-1] + (nk[k] + tau[k] * alpha) * ((nwz[w1, k] + beta) * (nwz[w2, k] + beta)) / ((2 * nk[k] + W * beta) * (2 * nk[k] + (W + 1) * beta))
	catch
		println(size(p))
		println(k)
		println(K)
		println(size(tau))
		println(size(nwz))
		println(size(nk))
		exit(1)
	end
	end

    p[K+1] = p[K] + (tau[K+1] * alpha / W)

	u = p[K+1]*rand()
	for k in 1:K+1
		if u < p[k]
            return k
		end
    end
end

function initialize(dictionary, biterms, W, Kmax, alpha, beta, gamma)
    K = 0
	topics = Array(Int64, length(biterms))
	nk = zeros(Int64, K)
    nwz = zeros(Int64, length(dictionary), K)
    U1 = Int64[]
    U0 = [1:Kmax]
    tau = 1
    for (i, biterm) in enumerate(biterms)

        k = sampletopicindex(nk, nwz, biterm[1], biterm[2], W, K, tau, alpha, beta)
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

                println("$_k (K = $K)")
                nk[_k] += 1
                nwz[biterm[1], _k] += 1
                nwz[biterm[2], _k] += 1
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

function gibbs(corpus, topics, nk, nwz, W, K, Kmax, U1, U0, tau, alpha, beta, gamma, iterations)
loop = 0
    KOverTime = Int64[]
	for n in 1:iterations
	try
		push!(KOverTime, K)
        println("Iteration $n")
		println(K)
		println(size(nwz))
		println(U1)
		println(U0)
		println(nk)
        for (i, biterm) in enumerate(biterms)
				if K > length(nk)
			print(loop)
			println(K)
			println(size(nk))
			println(size(nwz))
			println(U1)
			println(U0[1:10])
			exit(1)
		end
        loop = 0
            w1 = biterm[1]
            w2 = biterm[2]
            k = topics[i]
            
            nk[k] -= 1
            nwz[w1, k] -= 1
            nwz[w2, k] -= 1
            
            k = sampletopicindex(nk, nwz, w1, w2, W, K, tau, alpha, beta)
            
            if k > K
			loop = 1
                try
                    _k = shift!(U0)
                catch
                    U0 = [K+1:K*2]
                    _k = shift!(U0)
                end
                topics[i] = _k
                push!(U1, _k)
                if _k == K + 1
				loop = 2
                    push!(nk, 1)

                    _nwz = zeros(Int64, W)
                    _nwz[w1] += 1
                    _nwz[w2] += 1

                    nwz = [nwz _nwz]

                    tau = rand(Dirichlet(vec([nk .+ beta, gamma])), 1)
                else
				loop = 3
                    nk[_k] += 1
                    nwz[w1, _k] += 1
                    nwz[w2, _k] += 1
                end
                K += 1
            else
			loop = 4
                _k = U1[k]
                topics[i] = _k            
                nk[_k] += 1
                nwz[w1, _k] += 1
                nwz[w2, _k] += 1
            end
        end
        println("Pruning empty topics")
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
		U0 = [K+1:K+100]
		tau = rand(Dirichlet(vec([nk .+ beta, gamma])), 1)
		catch
			println(K)
			println(size(nk))
			println(size(nwz))
			println(U1)
			println(U0[1:10])
						print(loop)
			exit(1)
		end
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
    println(K)
    print(size(phi))
    for k = 1:K
        topic = hcat(phi'[:,k], dictionary)
        println("Topic $k:\n $(sortrows(topic, rev=true, by=x->(x[1]))[1:nwords,:])")
    end
end

using HDF5, JLD

(dictionary, biterms) = parseCorpus("data/reuters_vocab.dat", "data/reuters_train.dat", 15)
for gamma=5:5:100
    println("Running HDP for gamma=", gamma)
    (topics, nk, nwz, K, Kmax, U1, U0, tau) = initialize(dictionary, biterms, length(dictionary), 1000, 1, 0.1, max(gamma, 1))
    (topics, nk, nwz, K, Kmax, U1, U0, tau, KOverTime) = gibbs(biterms, topics, nk, nwz, length(dictionary), K, Kmax, U1, U0, tau, 1, 0.1, max(gamma, 1), 1000)
    save(string("models/bdp/bdp_1_01_",gamma), "Z", topics, "nk", nk, "nwz", nwz, "k_over_time", KOverTime)
end

#using ArgParse
#s = ArgParseSettings()
#@add_arg_table s begin
#    "--alpha", "-a"
#    help = "The alpha hyperparameter"
#    arg_type = Number
#    default = 1
#    "--beta", "-b"
#    help = "The beta hyperparameter"
#    arg_type = Number
#    default = 0.01
#    "--gamma", "-g"
#    help = "The gamma hyperparameter"
#    arg_type = Number
#    default = 1
#    "--iterations", "-i"
#    help = "The number of iterations"
#    arg_type = Int
#    required = true
#    "--dictionary", "-d"
#    help = "Path to vocabulary file"
#    "corpus"
#    help = "Path to corpus file"
#    required = true
#    "--window", "-w"
#    help = "The context window in which co-occurrences are considered"
#    arg_type = Int
#    default = 15
#end

#parsed_args = parse_args(s)
#println("Running BTM with the following settings:\n$parsed_args")
#(dictionary, biterms) = parseCorpus(parsed_args["dictionary"], parsed_args["corpus"], parsed_args["window"])
#(topics, nk, nwz, K, Kmax, U1, U0, tau) = initialize(dictionary, biterms, length(dictionary), parsed_args["iterations"], parsed_args["alpha"], parsed_args["beta"], parsed_args["gamma"])
#(topics, nk, nwz, K, Kmax, U1, U0, tau) = gibbs(biterms, topics, nk, nwz, length(dictionary), K, Kmax, U1, U0, tau, parsed_args["alpha"], parsed_args["beta"], parsed_args["gamma"], parsed_args["iterations"])
#printTopics(estimatePhi(nwz', nk, parsed_args["beta"], dictionary), dictionary, 10, K)



