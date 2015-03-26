function parseCorpus(dpath, cpath)
	window = 15
	dfile = open(dpath, "r")
	cfile = open(cpath, "r")

	dictionary = map(chomp, readlines(dfile))

	biterms = Array((Int64, Int64), 0)
	for (i, line) in enumerate(eachline(cfile))
		println("Parsing document $i")
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

function initTopics(dictionary, biterms, K)
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

function sampleTopic(nk, nwz, biterm, K, W, alpha, beta)
	p = zeros(K)
	p[1] = (nk[1] + alpha) * ((nwz[biterm[1], 1] + beta) * (nwz[biterm[2], 1] + beta)) / ((2 * nk[1] + W * beta) * (2 * nk[1] + (W + 1) * beta))
	for k = 2:K 
		p[k] = p[k-1] + (nk[k] + alpha) * ((nwz[biterm[1], k] + beta) * (nwz[biterm[2], k] + beta)) / ((2 * nk[k] + W * beta) * (2 * nk[k] + (W + 1) * beta))
	end
	u = p[K] * rand()
	for topic in 1:K
		if u < p[topic]
			return topic
		end
	end
end

function gibbs(dictionary, biterms, topics, nk, nwz, K, alpha, beta, iterations)
	W = length(dictionary)
	for n = 1:iterations
		println("Iteration $n")
		println("Topic counts: ", nk)
		for (i, biterm) in enumerate(biterms)
			topic = topics[i]
			nk[topic] -= 1
			nwz[biterm[1], topic] -= 1
			nwz[biterm[2], topic] -= 1

			topic = sampleTopic(nk, nwz, biterm, K, W, alpha, beta)

			topics[i] = topic
			nk[topic] += 1
			nwz[biterm[1], topic] += 1
			nwz[biterm[2], topic] += 1
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

function printTopics(phi, dictionary, nwords)
	println([dictionary phi'])
end

function exampleRun()
	(dictionary, biterms) = parseCorpus("data/vocab.txt", "data/ap.dat")
	topics, nk, nwz = gibbs(dictionary, biterms, initTopics(dictionary, biterms, 5)..., 5, 1, 0.1, 100)
	nothing
end

exampleRun()