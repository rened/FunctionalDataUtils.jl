println("Starting runtests.jl $(join(ARGS, " ")) ...")
using FunctionalDataUtils, FactCheck, Compat
FactCheck.setstyle(:compact)

shouldtest(f, a) = length(ARGS) == 0 || in(a, ARGS) ? facts(f, a) : nothing
shouldtestcontext(f, a) = length(ARGS) < 2 || a == ARGS[2] ? facts(f, a) : nothing

shouldtest("machinelearning") do
    shouldtestcontext("randbase") do 
        # @test_equal size(randbase(2,10)) (2,10)
        # @test_almostequal sqrt(sum(randbase(2,10).^2, 2)) [1. 1.]'
        # @test_almostequal (@p randbase 2 10  | (.^) _ 2 |  sum _ 2) [1. 1.]'
    end
end

shouldtest("computing") do
    shouldtestcontext("fasthash") do
        @fact fasthash(1) --> "20eae029a26a15420f9ed6d7a9999e823f251b9d44adea5a504e1d8a4f7c5512"
        @fact fasthash(1.) --> "cb5415fc791f65b70b303678e3601d60ac527cc4cdd3a10d1f85b7423b270112"
        @fact fasthash('a') --> "02bd3cf2fa502d7ed9fbd7b4c69b6699d03161d6dfde5d0868277429ad7cabb6"
        @fact fasthash("a") --> "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        @fact fasthash(["a"]) --> "cd372fb85148700fa88095e3492d3f9f5beb43e555e5ff26d95f5a6adc36f8e6"
        @fact fasthash([1]) --> "7c9fa136d4413fa6173637e883b6998d32e1d675f88cddff9dcbcf331820f4b8"
        @fact fasthash(Uint8[1]) --> "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        @fact fasthash(Int8[1]) --> "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        @fact fasthash(Uint16[1]) --> "47dc540c94ceb704a23875c11273e16bb0b8a87aed84de911f2133568115f254"
        @fact fasthash(Int16[1]) --> "47dc540c94ceb704a23875c11273e16bb0b8a87aed84de911f2133568115f254"
        @fact fasthash(Uint32[1]) --> "67abdd721024f0ff4e0b3f4c2fc13bc5bad42d0b7851d456d88d203d15aaa450"
        @fact fasthash(Int32[1]) --> "67abdd721024f0ff4e0b3f4c2fc13bc5bad42d0b7851d456d88d203d15aaa450"
        @fact fasthash(Uint64[1]) --> "7c9fa136d4413fa6173637e883b6998d32e1d675f88cddff9dcbcf331820f4b8"
        @fact fasthash(Int64[1]) --> "7c9fa136d4413fa6173637e883b6998d32e1d675f88cddff9dcbcf331820f4b8"
        @fact fasthash(Any[1,"test"]) --> "b59f20a7e3513ef702971ec5cbd288f19d760df46d49d5d27edb82fe73fc257b"
        data = Any[1,1., 'a', "a", rand(10,200), zeros(1000,10000), @compat Dict("a" => 1, 2 => "b")]
        r = [fasthash(x) for x in data]
        @fact all(r .!= nothing) --> true   # just to avoid showing "0 facts verified". we did survive the above, after all.
        @fact fasthash(1:2) --> "48adce367d9daa9d76033a4693b20d7fab23d357bf97e49a036f2dd32b1afff4"
        @fact fasthash(1:3:10) --> "dc29cadf1701b9023b1170a8101b40928770dbbd967e2740afa4986499522ca3"
    end
end

shouldtest("computervision") do
    shouldtestcontext("iimg") do
        a = [1 2 3]
        @fact size(iimg(a)) --> (1,3)
        @fact typeof(iimg(a)) --> Array{Float64,2}
        @fact iimg(a) --> [1 3 6]
        @fact iimg([1 2 3; 4 5 6]) --> [1.0 3.0 6.0; 5.0 12.0 21.0]
        @fact iimg(ones(1,2,3)) --> float(cat(3,[1 2],[2 4],[3 6]))
    end

    shouldtestcontext("interp3") do
        a = [o for m in 1:2, n in 1:2, o in 1:2]
        @fact interp3(a,1.,1.,1.5) --> 1.5 
        @fact interp3(a,2.,2.,1.5) --> 1.5
        a = [n for m in 1:2, n in 1:2, o in 1:2]
        @fact interp3(a,1.,1.5,1.) --> 1.5
        @fact interp3(a,2.,1.5,2.) --> 1.5
        a = [m for m in 1:2, n in 1:2, o in 1:2]
        @fact interp3(a,1.5,1.,1.) --> 1.5
        @fact interp3(a,1.5,2.,2.) --> 1.5
    end

    shouldtestcontext("interp3with01coords") do
        a = [o for m in 1:2, n in 1:2, o in 1:2]
        @fact interp3with01coords(a,0.,0.,0.5) --> 1.5
        @fact interp3with01coords(a,1.,1.,0.5) --> 1.5
        a = [n for m in 1:2, n in 1:2, o in 1:2]
        @fact interp3with01coords(a,0.,0.5,0.) --> 1.5
        @fact interp3with01coords(a,1.,0.5,1.) --> 1.5
        a = [m for m in 1:2, n in 1:2, o in 1:2]
        @fact interp3with01coords(a,0.5,0.,0.) --> 1.5
        @fact interp3with01coords(a,0.5,1.,1.) --> 1.5
    end

    shouldtestcontext("resize") do
        @fact resize(ones(Int,1,1), [1, 1]) --> ones(Int, 1, 1)
        @fact resize(ones(Float64,1,1), 1) --> ones(Float64, 1, 1)
        @fact resize(ones(Float64,1,1,1), [1, 1, 1]) --> ones(Float64, 1, 1, 1)
        @fact resize(ones(Int,1,1), [1, 10]) --> ones(Int, 1, 10)
        @fact maximum(abs(resize([1. 2.], [1, 4])-[1. 1.33333 1.66666 2.]))<0.01 --> true
        @fact resize([1. 3.], [1, 3]) --> [1. 2. 3.]
    end

    shouldtestcontext("bwlabel") do
        img = [1 2 3]
        @fact bwlabel(img) --> img

        img = [1 2 3 0 5 6]
        @fact bwlabel(img) --> [1 2 3 1 5 6]
        @fact bwlabel(img,4) --> [1 2 3 4 5 6]

        img = [1 1 0 5 2 2; 3 3 0 0 4 4]
        bwlabel(img)
    end

    shouldtestcontext("border") do
        img = zeros(5,7)
        img[3,4] = 1
        @fact border(img)  -->  img
        img[2:4,3:6] = 1
        b = copy(img)
        b[3,4:5] = 0
        @fact border(img)  -->  b
    end
    
    shouldtestcontext("bwdist") do
        a = [0 0 0 1 0 0]
        @fact bwdist(a)  -->  [3 2 1 0 1 2]
        @fact bwdist([1 10;2 12], [1 10; 1 10])  -->  [1,2]
    end

    shouldtestcontext("rle") do
        data = [1 1 2 2 2 3 3 3 3]
        @fact unrle(rle(data)) --> data
        data = rand(1:5, 1, 1000)
        @fact unrle(rle(data)) --> data
        for t = [Int16, Int32, Int64, Float32, Float64]
            for s1 = 1:10, s2 = 1:10
                data = rand(t, s1, s2)
                @test_equal unrle(rle(data)) data
            end
        end
    end

    shouldtestcontext("resizeminmax") do
        resize(a) = size(resizeminmax(a, [30,40], [130, 140]))
        @fact resize(ones(1,1)) --> (30,40) 
        @fact resize(ones(100,200)) --> (85,140) 
        @fact resize(ones(10,200)) --> (30,140) 
        @fact resize(ones(50,10)) --> (125,40) 
        @fact resize(ones(50,60)) --> (50,60) 
        @fact resize(ones(1,600)) --> (30,140) 
        @fact resize(ones(600,1)) --> (130,40) 
    end

    shouldtestcontext("blocks") do
        a = ones(16,32)
        ind = stridedblocksub(a, 8; keepshape = true)
        @fact size(ind) --> (2,4)
        ind = stridedblocksub(a, 8)
        @fact len(ind) --> 8
        ind =  stridedblocksub(a, 8, 1)
        @fact len(ind) --> 9*25
        a = flatten(Any[ones(2,2) 2*ones(2,2); 3*ones(2,2) 4*ones(2,2)])
        subs = stridedblocksub(a, 2)
    end

    shouldtestcontext("meshgrid") do
        @fact meshgrid(2,3)  -->  [1 2 1 2 1 2; 1 1 2 2 3 3]
        @fact meshgrid(1:2,1:3)  -->  [1 2 1 2 1 2; 1 1 2 2 3 3]
        @fact meshgrid(2:3,2:4)  -->  [2 3 2 3 2 3; 2 2 3 3 4 4]
        @fact meshgrid(2:3,[2, 0, 4])  -->  [2 3 2 3 2 3; 2 2 0 0 4 4]
        @fact centeredmeshgrid(2:3,2:4)  -->  [0 1 0 1 0 1; -1 -1 0 0 1 1]
    end

    shouldtestcontext("inpointcloud") do
        l = row(linspace(0,1,100))
        o = row(ones(100))
        shape2d = [l o flipdim(l,2) 0*o; 0*o l o flipdim(l,2)]*10.+[2.5;5.5]
        shouldbe2d = zeros(30,30)
        shouldbe2d[3:12,6:15] = 1
        coords = meshgrid(30,30)
        result2d = reshape(map(coords, x->inpointcloud(x, shape2d)), 30,30);
        @fact result2d  -->  shouldbe2d

        plate1 = [meshgrid(1:10,1:10); zeros(1,100)] 
        plate2 = [meshgrid(1:10,1:10); 10*ones(1,100)] 
        plate3 = plate1[[1, 3, 2],:]
        plate4 = plate2[[1, 3, 2],:]
        plate5 = plate1[[3, 1, 2],:]
        plate6 = plate2[[3, 1, 2],:]
        shape3d = [plate1 plate2 plate3 plate4 plate5 plate6].+[2.5;5.5;5.5]

        shouldbe3d = zeros(30,30,30)
        shouldbe3d[3:12,6:15,6:15] = 1
        coords = meshgrid(30,30,30)
        result3d = reshape(map(coords, x->inpointcloud(x, shape3d)), 30,30,30);
        result3d == shouldbe3d
        @fact sum(result3d-shouldbe3d)  -->  38
    end

    shouldtestcontext("sampler") do
        image = rand(200,100)
        sampler = makeSampler(image, 32)
        coords = asint(50 + 50*randn(2,100))
        if VERSION.minor >= 4
            @fact size(@p map coords sampler) --> (32,32,100)
        else
            @fact size(@p map coords x->call(sampler,x)) --> (32,32,100)
        end
        out = zeros(1024,1)
        pos = col([50,50])
        @fact size(sample!(out, pos, sampler)) --> (1024,1)

        sampler = makeSampler(image, 32, col = true)
        @fact size(sample(pos, sampler)) --> (1024,1)

        sampler = makeSampler(image, 32, col = true, normmeanstd = true)
        r = sample(pos, sampler)
        @fact abs(mean(r))  < 1e-5 --> true
        @fact abs(std(r)-1) < 1e-5 --> true

        image = [1 2 3 4 5; 10 20 30 40 50; 11 22 33 44 55]
        sampler = makeSampler(image, 3, centered = false)
        @fact sample(col([1,2]), sampler) --> [2 3 4; 20 30 40; 22 33 44]
        sampler = makeSampler(image, 3, centered = true)
        @fact sample(col([1,2]), sampler) --> [1 2 3; 10 20 30; 11 22 33]
        @fact sample(col([1,3]), sampler) --> [2 3 4; 20 30 40; 22 33 44]

        image = ones(20,10)
        sampler = makeSampler(image, 3, centered = false)
        sample(sampler,ones(siz(image)))
        sample(sampler,siz(image))

        sampler = makeSampler(image, 3, centered = true)
        sample(sampler,ones(siz(image)))
        sample(sampler,siz(image))
    end
end

shouldtest("graphics") do
    shouldtestcontext("jetcolormap") do
        r = jetcolormap(10)
        @fact all(r .>= 0) --> true
        @fact all(r .<= 1) --> true
    end

    shouldtestcontext("asimagesc") do
        img = [1 2 3 4 5 6 7 8 9]
        r = asimagesc(img)
        @fact size(r,3) --> 3
    end    

    shouldtestcontext("blocksvisu") do
        @fact blocksvisu([1 2 3]) --> [1 0 3; 0 0 0; 2 0 0]
        @fact blocksvisu([1 2 3; 1 2 3; 1 2 3; 1 2 3]) --> [1 1 0 3 3; 1 1 0 3 3; 0 0 0 0 0; 2 2 0 0 0; 2 2 0 0 0]
    end

    shouldtestcontext("overlaygradient") do
        img = rand(100,200)
        sp = zeros(size(img))
        sp[25:75, 50:150] = 1
        @fact size(overlaygradient(img, sp)) --> (100,200,3)
    end 
 end

shouldtest("numerical") do
    shouldtestcontext("nanfunctions") do
        @fact nanmedian([1 2 3], 1)  -->  [1.0 2.0 3.0]
        @fact isequal(nanmedian([1 2 NaN], 1),  [1.0 2.0 NaN]) --> true
        @fact nanmedian([1 2 3], 2)  -->  row([2.0])
        @fact nanmedian([1 2 NaN], 2)  -->  row([1.5])
        @fact nanmedian([1, 2, NaN])  -->  1.5
    end
    shouldtestcontext("distance") do
        @fact distance(1,1)  -->  0
        @fact distance(1.,1.)  -->  0.
        @fact distance([1,2,3],[2,2,3])  -->  1
        @fact distance([0 1; 0 1],[0 1; 0 1])  -->  [0. sqrt(2); sqrt(2) 0.]
    end
    shouldtestcontext("norms") do
        @fact normsum([1 2 1])  -->  [1/4 1/2 1/4]
        @fact norm01([1 2 3])  -->  [0 1/2 1]
        @fact normeuclid([1 1])  -->  1./[sqrt(2) sqrt(2)]
        @fact normmeanstd([1 2 3])  -->  [-1 0 1]
    end
    shouldtestcontext("extrema") do
        @fact (@p maximum [1 2 3; 4 5 0] (x->x[2]))  -->  [2 5]'
        @fact (@p minimum [1 2 3; 4 5 0] (x->x[2]))  -->  [3 0]'
    end
    shouldtestcontext("valuemap") do
        @fact valuemap([0 1 2 3], [10 20 30])  -->  [0. 10. 20. 30.]
    end
    shouldtestcontext("clamp") do
        @fact clamp(1,1,1)  -->  1
        @fact clamp(0,1,2)  -->  1
        @fact clamp(2,1,2)  -->  2
        @fact clamp([-1 2 3 10; -5 10 11  100],[1 10],[3 15])  -->  [1 2 3 3; 10 10 11 15]
        @fact clamp(Float32[-1 2 3 10; -5 10 11  100],[1 10],[3 15])  -->  [1 2 3 3; 10 10 11 15]
        @fact clamp([1,2],[3,4],[5,6])   -->  [3,4]
        @fact clamp([0 1 11; -5 3 20], rand(10,15))  -->  [1 1 10; 1 3 15]
        @fact clamp([0 1 11; -5 3 20; -4 3 5], rand(10,15,4))  -->  [1 1 10; 1 3 15; 1 3 4]
    end
    shouldtestcontext("rand") do
        @fact size(randsiz(siz(rand(2,3)))) --> (2,3)
        @fact eltype(randsiz([2 3]', Float32)) --> Float32
    end
    shouldtestcontext("plus") do
        @fact (@p plus zeros(2,3) ones(2,1))  -->  ones(2,3)
        @fact (@p minus zeros(2,3) ones(2,1))  -->  -ones(2,3)
        @fact (@p times ones(2,3) zeros(2,1))  -->  zeros(2,3)
        @fact (@p divby ones(2,3) 2*ones(2,1))  -->  0.5*ones(2,3)
    end
end

exitwithteststatus()
