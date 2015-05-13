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

function initialize(dictionary::Array{ASCIIString, 1}, biterms::Array{(Int64, Int64), 1}, K::Int64)
	topics = Array(Int64, length(biterms))
	nk = zeros(Int64, K)
	nwz = zeros(Int64, length(dictionary), K)
	for (i, biterm) in enumerate(biterms)
		topic = rand(1:K)
		topics[i] = topic
		nk[topic] += 1
		nwz[biterm[1], topic] += 1
		nwz[biterm[2], topic] += 1
	end
	topics, nk, nwz
end

function sampleTopic(nk::Array{Int64,1}, nwz::Array{Int64,2}, w1::Int64, w2::Int64, K::Int64, W::Int64, alpha::Number, beta::Number)
	p = zeros(K)
	p[1] = (nk[1] + alpha) * ((nwz[w1, 1] + beta) * (nwz[w2, 1] + beta)) / ((2 * nk[1] + W * beta) * (2 * nk[1] + (W + 1) * beta))
	for k = 2:K 
		p[k] = p[k-1] + (nk[k] + alpha) * ((nwz[w1, k] + beta) * (nwz[w2, k] + beta)) / ((2 * nk[k] + W * beta) * (2 * nk[k] + (W + 1) * beta))
	end
	u = p[K] * rand()
	for topic in 1:K
		if u < p[topic]
			return topic
		end
	end
end

function gibbs(dictionary::Array{ASCIIString, 1}, biterms::Array{(Int64, Int64), 1}, topics::Array{Int64,1}, nk::Array{Int64,1}, nwz::Array{Int64,2}, K::Int64, alpha::Number, beta::Number, iterations::Int64)
	W = length(dictionary)
	for n = 1:iterations
		println("Iteration $n")
        for (i, biterm) in enumerate(biterms)
            w1 = biterm[1]
            w2 = biterm[2]
			topic = topics[i]
			nk[topic] -= 1
			nwz[w1, topic] -= 1
			nwz[w2, topic] -= 1

			topic = sampleTopic(nk, nwz, w1, w2, K, W, alpha, beta)

			topics[i] = topic
			nk[topic] += 1
			nwz[w1, topic] += 1
			nwz[w2, topic] += 1
		end
	end
	topics, nk, nwz
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
Z, nk, nwz = initialize(dictionary, biterms, parsed_args["num_topics"])
gibbs(dictionary, biterms, Z, nk, nwz, parsed_args["num_topics"], parsed_args["alpha"], parsed_args["beta"], parsed_args["iterations"])

printTopics(estimatePhi(nwz', nk, parsed_args["beta"], dictionary), dictionary, 10, parsed_args["num_topics"])


