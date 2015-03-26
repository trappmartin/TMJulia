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

function initTopics(dictionary, corpus, K)
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
		println("Topic counts: ", nk)
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

function printTopics(phi, dictionary, nwords)
	println([dictionary phi'])
end

function exampleRun()
	(dictionary, corpus) = parseCorpus("data/vocab.txt", "data/ap.dat")
	alpha = 1
	beta = 0.1
	K = 5
	iterations = 100
	Z, ndk, nkw, nk = gibbs(dictionary, corpus, initTopics(dictionary, corpus, 5)..., K, alpha, beta, iterations)
	printTopics(estimatePhi(nkw, nk, beta, dictionary), dictionary, 10)

	nothing
end

exampleRun()