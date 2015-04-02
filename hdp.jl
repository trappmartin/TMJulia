using Distributions

function parsecorpus(dpath, cpath)
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
			topics[d][w] = k

			if k > K
				try
					_k = shift!(U0)
				catch
					U0 = [Kmax+1:Kmax*2]
					Kmax *= 2
					_k = shift!(U0)
				end

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
				ndk[d, _k] += 1
				nd[d] += 1
				nkw[_k, word] += 1
				nk[_k] += 1
			end

			ndk[d, k] += 1
			nd[d] += 1
			nkw[k, word] += 1
			nk[k] += 1
		end
	end


	topics, ndk, nd, nkw, nk, K, Kmax, U1, U0, tau
end

function gibbs(corpus, topics, ndk, nd, nkw, nk, W, K, Kmax, U1, U0, tau, alpha, beta, gamma, iterations)
	for n in 1:iterations
		println("Iteration $n")
		for (d, doc) in enumerate(corpus)
			for (w, word) in enumerate(doc)
				k = topics[d][w]
				ndk[d, k] -= 1
				nd[d] -= 1
				nkw[k, word] -= 1
				nk[k] -= 1

				k = sampletopicindex(ndk, nd, nkw, nk, d, word, W, K, tau, alpha, beta)

				topics[d][w] = k
				
				if k > K
					try
						_k = shift!(U0)
					catch
						U0 = [Kmax+1:Kmax*2]
						Kmax *= 2
						_k = shift!(U0)
					end

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
					ndk[d, _k] += 1
					nd[d] += 1
					nkw[_k, word] += 1
					nk[_k] += 1
				end
			end
		end

		for k in 1:K
			if nk[k] == 0
				delete!(U1, k)
				unshift!(U0, k)
				ndk[:,k] = 0
				nkw[k, :] = 0
				K -= 1
			end
		end 
		tau = rand(Dirichlet(vec([nk .+ beta, gamma])), 1)
	end
end
(dictionary, corpus) = parsecorpus("data/vocab.txt", "data/ap.dat")
	
(topics, ndk, nd, nkw, nk, K, Kmax, U1, U0, tau) = initialize(dictionary, corpus, length(dictionary), 100, 1, 0.001, 10)
gibbs(corpus, topics, ndk, nd, nkw, nk, length(dictionary), K, Kmax, U1, U0, tau, 1, 0.001, 10, 100)