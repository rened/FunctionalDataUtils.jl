# FIXME see whether we can remove the broadcast calls required for 0.7

println("\n\n\nStarting runtests.jl $(join(ARGS, " ")) ...")

using FunctionalData, FunctionalDataUtils, Test #, Colors
using Statistics

macro shouldtestset(a,b) length(ARGS) < 1 || ARGS[1] == a ?  :(@testset $a $b) : nothing end
macro shouldtestset2(a,b) length(ARGS) < 2 || ARGS[2] == a ?  :(@testset $a $b) : nothing end

@shouldtestset "machinelearning" begin
    @shouldtestset2 "randbase" begin
        # @test_equal size(randbase(2,10)) (2,10)
        # @test_almostequal sqrt(sum(randbase(2,10).^2, 2)) [1. 1.]'
        # @test_almostequal (@p randbase 2 10  | (.^) _ 2 |  sum _ 2) [1. 1.]'
    end
    @shouldtestset2 "randproj" begin
    end
    @shouldtestset2 "loocv" begin
    end
end

@shouldtestset "computing" begin
    @shouldtestset2 "fasthash" begin
        @test fasthash(1) == "20eae029a26a15420f9ed6d7a9999e823f251b9d44adea5a504e1d8a4f7c5512"
        @test fasthash(1.) == "cb5415fc791f65b70b303678e3601d60ac527cc4cdd3a10d1f85b7423b270112"
        @test fasthash('a') == "94923c7a4899a5baf82f29551f5e4652e3b7ee50699bbd0b4ddba1538e5963b2"
        @test fasthash("a") == "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
        @test fasthash(["a"]) == "da3811154d59c4267077ddd8bb768fa9b06399c486e1fc00485116b57c9872f5"
        @test fasthash([1]) == "7c9fa136d4413fa6173637e883b6998d32e1d675f88cddff9dcbcf331820f4b8"
        @test fasthash(UInt8[1]) == "4bf5122f344554c53bde2ebb8cd2b7e3d1600ad631c385a5d7cce23c7785459a"
        @test fasthash(Int8[1]) == "4bf5122f344554c53bde2ebb8cd2b7e3d1600ad631c385a5d7cce23c7785459a"
        @test fasthash(UInt16[1]) == "47dc540c94ceb704a23875c11273e16bb0b8a87aed84de911f2133568115f254"
        @test fasthash(Int16[1]) == "47dc540c94ceb704a23875c11273e16bb0b8a87aed84de911f2133568115f254"
        @test fasthash(UInt32[1]) == "67abdd721024f0ff4e0b3f4c2fc13bc5bad42d0b7851d456d88d203d15aaa450"
        @test fasthash(Int32[1]) == "67abdd721024f0ff4e0b3f4c2fc13bc5bad42d0b7851d456d88d203d15aaa450"
        @test fasthash(UInt64[1]) == "7c9fa136d4413fa6173637e883b6998d32e1d675f88cddff9dcbcf331820f4b8"
        @test fasthash(Int64[1]) == "7c9fa136d4413fa6173637e883b6998d32e1d675f88cddff9dcbcf331820f4b8"
        @test fasthash(Any[1,"test"]) == "b59f20a7e3513ef702971ec5cbd288f19d760df46d49d5d27edb82fe73fc257b"
        data = Any[1,1., 'a', "a", rand(10,200), zeros(1000,10000), Dict("a" => 1, 2 => "b")]
        r = [fasthash(x) for x in data]
        @test fasthash(1:2) == "48adce367d9daa9d76033a4693b20d7fab23d357bf97e49a036f2dd32b1afff4"
        @test fasthash(1:3:10) == "dc29cadf1701b9023b1170a8101b40928770dbbd967e2740afa4986499522ca3"
    end
end

@shouldtestset "computervision" begin
    @shouldtestset2 "warp" begin
        img = rand(10,20,3)
        a = rand(3:9,2,4)
        b = copy(a)
        b[:,end] = 10
        warp(img,a,b)
        @test size(warp(img,a,b))  ==  (10,20,3)
        @test size(warp(img,a,b,(8,9)))  ==  (8,9,3)

    end
    @shouldtestset2 "iimg" begin
        a_ = [1 2 3]
        @test size(iimg(a_)) == (1,3)
        @test typeof(iimg(a_)) == Array{Float64,2}
        @test iimg(a_) == [1 3 6]
        @test iimg([1 2 3; 4 5 6]) == [1.0 3.0 6.0; 5.0 12.0 21.0]
        @test iimg(ones(1,2,3)) == float(cat([1 2],[2 4],[3 6],dims=3))
    end

    @shouldtestset2 "interp3" begin
        a = [o for m in 1:2, n in 1:2, o in 1:2]
        @test interp3(a,1.,1.,1.5) == 1.5 
        @test interp3(a,2.,2.,1.5) == 1.5
        a = [n for m in 1:2, n in 1:2, o in 1:2]
        @test interp3(a,1.,1.5,1.) == 1.5
        @test interp3(a,2.,1.5,2.) == 1.5
        a = [m for m in 1:2, n in 1:2, o in 1:2]
        @test interp3(a,1.5,1.,1.) == 1.5
        @test interp3(a,1.5,2.,2.) == 1.5
    end

    @shouldtestset2 "interp3with01coords" begin
        a = [o for m in 1:2, n in 1:2, o in 1:2]
        @test interp3with01coords(a,0.,0.,0.5) == 1.5
        @test interp3with01coords(a,1.,1.,0.5) == 1.5
        a = [n for m in 1:2, n in 1:2, o in 1:2]
        @test interp3with01coords(a,0.,0.5,0.) == 1.5
        @test interp3with01coords(a,1.,0.5,1.) == 1.5
        a = [m for m in 1:2, n in 1:2, o in 1:2]
        @test interp3with01coords(a,0.5,0.,0.) == 1.5
        @test interp3with01coords(a,0.5,1.,1.) == 1.5
    end

    @shouldtestset2 "resize" begin
        @test resize(ones(Int,1,1), [1, 1]) == ones(Int, 1, 1)
        @test resize(ones(Float64,1,1), 1) == ones(Float64, 1, 1)
        @test resize(ones(Float64,1,1,1), [1, 1, 1]) == ones(Float64, 1, 1, 1)
        @test resize(ones(Int,1,1), [1, 10]) == ones(Int, 1, 10)
        @test maximum(abs.(resize([1. 2.], [1, 4])-[1. 1.33333 1.66666 2.])) < 0.01
        @test resize([1. 3.], [1, 3]) == [1. 2. 3.]
    end

    @shouldtestset2 "bwlabel" begin
        img = [1 2 3]
        @test bwlabel(img) == img

        img = [1 2 3 0 5 6]
        @test bwlabel(img) == [1 2 3 1 5 6]
        @test bwlabel(img,4) == [1 2 3 4 5 6]

        img = [1 1 0 5 2 2; 3 3 0 0 4 4]
        bwlabel(img)
    end

    @shouldtestset2 "border" begin
        img = zeros(5,7)
        img[3,4] = 1
        @test border(img)  ==  img
        img[2:4,3:6] = 1
        b = copy(img)
        b[3,4:5] = 0
        @test border(img)  ==  b
    end

    @shouldtestset2 "bwdist" begin
        a = [0 0 0 1 0 0]
        @test bwdist(a)  ==  [3 2 1 0 1 2]
        @test bwdist([1 10;2 12], [1 10; 1 10])  ==  [1,2]
    end

    @shouldtestset2 "rle" begin
        data = [1 1 2 2 2 3 3 3 3]
        @test unrle(rle(data)) == data
        data = rand(1:5, 1, 1000)
        @test unrle(rle(data)) == data
        for t = [Int16, Int32, Int64, Float32, Float64]
            for s1 = 1:10, s2 = 1:10
                data = rand(t, s1, s2)
                @test unrle(rle(data)) == data
            end
        end
    end

    @shouldtestset2 "resizeminmax" begin
        f(a) = size(resizeminmax(a, [30,40], [130, 140]))
        @test f(ones(1,1)) == (30,40) 
        @test f(ones(100,200)) == (85,140) 
        @test f(ones(10,200)) == (30,140) 
        @test f(ones(50,10)) == (125,40) 
        @test f(ones(50,60)) == (50,60) 
        @test f(ones(1,600)) == (30,140) 
        @test f(ones(600,1)) == (130,40) 
    end

    @shouldtestset2 "blocks" begin
        a = ones(16,32)
        ind = stridedblocksub(a, 8; keepshape = true)
        @test size(ind) == (2,4)
        ind = stridedblocksub(a, 8)
        @test len(ind) == 8
        ind =  stridedblocksub(a, 8, 1)
        @test len(ind) == 9*25
        a = flatten(Any[ones(2,2) 2*ones(2,2); 3*ones(2,2) 4*ones(2,2)])
        subs = stridedblocksub(a, 2)
    end

    @shouldtestset2 "meshgrid" begin
        @test meshgrid(2,3)  ==  [1 2 1 2 1 2; 1 1 2 2 3 3]
        @test meshgrid(1:2,1:3)  ==  [1 2 1 2 1 2; 1 1 2 2 3 3]
        @test meshgrid(2:3,2:4)  ==  [2 3 2 3 2 3; 2 2 3 3 4 4]
        @test meshgrid(2:3,[2, 0, 4])  ==  [2 3 2 3 2 3; 2 2 0 0 4 4]
        @test centeredmeshgrid(2:3,2:4)  ==  [0 1 0 1 0 1; -1 -1 0 0 1 1]
    end

    # FIXME reenable once MultivariateStats works on 0.7
    # @shouldtestset2 "inpointcloud" begin
    #     l = row(linspace(0,1,100))
    #     o = row(ones(100))
    #     shape2d = [l o flipdim(l,2) 0*o; 0*o l o flipdim(l,2)]*10 .+ [2.5;5.5]
    #     shouldbe2d = zeros(30,30)
    #     shouldbe2d[3:12,6:15] = 1
    #     coords = meshgrid(30,30)
    #     result2d = reshape(map(coords, x->inpointcloud(x, shape2d)), 30,30);
    #     @test result2d  ==  shouldbe2d

    #     plate1 = [meshgrid(1:10,1:10); zeros(1,100)] 
    #     plate2 = [meshgrid(1:10,1:10); 10*ones(1,100)] 
    #     plate3 = plate1[[1, 3, 2],:]
    #     plate4 = plate2[[1, 3, 2],:]
    #     plate5 = plate1[[3, 1, 2],:]
    #     plate6 = plate2[[3, 1, 2],:]
    #     shape3d = [plate1 plate2 plate3 plate4 plate5 plate6] .+ [2.5;5.5;5.5]

    #     shouldbe3d = zeros(30,30,30)
    #     shouldbe3d[3:12,6:15,6:15] = 1
    #     coords = meshgrid(30,30,30)
    #     result3d = reshape(map(coords, x->inpointcloud(x, shape3d)), 30,30,30);
    #     result3d == shouldbe3d
    #     @test sum(result3d-shouldbe3d)  ==  38
    # end

    @shouldtestset2 "sampler" begin
        image = rand(200,100)
        sampler = makeSampler(image, 32)
        coords = asint(broadcast(+, 50, 50*randn(2,100)))
        @test size(@p map coords sampler) == (32,32,100)
        out = zeros(1024,1)
        pos = col([50,50])
        @test size(sample!(out, pos, sampler)) == (1024,1)

        sampler = makeSampler(image, 32, col = true)
        @test size(sample(pos, sampler)) == (1024,1)

        sampler = makeSampler(image, 32, col = true, normmeanstd = true)
        r = sample(pos, sampler)
        @test abs(mean(r))  < 1e-5
        @test abs(std(r)-1) < 1e-5

        image = [1 2 3 4 5; 10 20 30 40 50; 11 22 33 44 55]
        sampler = makeSampler(image, 3, centered = false)
        @test sample(col([1,2]), sampler) == [2 3 4; 20 30 40; 22 33 44]
        sampler = makeSampler(image, 3, centered = true)
        @test sample(col([1,2]), sampler) == [1 2 3; 10 20 30; 11 22 33]
        @test sample(col([1,3]), sampler) == [2 3 4; 20 30 40; 22 33 44]

        image = ones(20,10)
        sampler = makeSampler(image, 3, centered = false)
        sample(sampler, col([1 1]))
        sample(sampler,siz(image))

        sampler = makeSampler(image, 3, centered = true)
        sample(sampler, col([1 1]))
        sample(sampler,siz(image))
    end
end

# @shouldtestset "graphics" begin
#     @shouldtestset2 "jetcolormap" begin
#         r = jetcolormap(10)
#         @test all(broadcast(>=, r, 0)) == true
#         @test all(broadcast(<=, r, 1)) == true
#     end

#     @shouldtestset2 "jetcolors" begin
#         @test jetcolors(1:3) ≈ [0.0 0.5198019801980198 0.5396039603960396; 0.0 1.0 0.0; 0.5396039603960396 0.4801980198019802 0.0]
#         @test jetcolors(1:3,2,4) ≈ [0.0 0.0 0.5198019801980198; 0.0 0.0 1.0; 0.5396039603960396 0.5396039603960396 0.4801980198019802]
#         # @test jetcolorants(1:3) == Any[RGB{Float64}(0.0,0.0,0.5396039603960396),RGB{Float64}(0.5198019801980198,1.0,0.4801980198019802),RGB{Float64}(0.5396039603960396,0.0,0.0)] # disabled until Colors works on 0.7
#     end

#     @shouldtestset2 "asimagesc" begin
#         img = [1 2 3 4 5 6 7 8 9]
#         r = asimagescrgb(img)
#         @test size(r,3) == 3
#     end    

#     @shouldtestset2 "blocksvisu" begin
#         @test blocksvisu([1 2 3]) == [1 0 3; 0 0 0; 2 0 0]
#         @test blocksvisu([1 2 3; 1 2 3; 1 2 3; 1 2 3]) == [1 1 0 3 3; 1 1 0 3 3; 0 0 0 0 0; 2 2 0 0 0; 2 2 0 0 0]
#     end

#     @shouldtestset2 "overlaygradient" begin
#         img = rand(100,200)
#         sp = zeros(size(img))
#         sp[25:75, 50:150] = 1
#         @test size(overlaygradient(img, sp)) == (100,200,3)
#     end 
# end

@shouldtestset "numerical" begin
    @shouldtestset2 "nanfunctions" begin
        @test nanmedian([1 2 3], 1)  ==  [1.0 2.0 3.0]
        @test isequal(nanmedian([1 2 NaN], 1),  [1.0 2.0 NaN]) == true
        @test nanmedian([1 2 3], 2)  ==  row([2.0])
        @test nanmedian([1 2 NaN], 2)  ==  row([1.5])
        @test nanmedian([1, 2, NaN])  ==  1.5
    end
    @shouldtestset2 "distance" begin
        @test distance(1,1)  ==  0
        @test distance(1.,1.)  ==  0.
        @test distance([1,2,3],[2,2,3])  ==  1
        @test distance([0 1; 0 1],[0 1; 0 1])  ==  [0. sqrt(2); sqrt(2) 0.]
    end
    @shouldtestset2 "norms" begin
        @test normsum([1 2 1])  ==  [1/4 1/2 1/4]
        @test norm01([1 2 3])  ==  [0 1/2 1]
        @test normeuclid([1 1])  ==  broadcast(/, 1, [sqrt(2) sqrt(2)])
        @test normmeanstd([1 2 3])  ==  [-1 0 1]
        @test normquantile(collect(0:10))  ≈  asfloat64(collect(0:10))/10
        @test normquantile(collect(0:2:20))  ≈  asfloat64(collect(0:10))/10
    end
    @shouldtestset2 "extrema" begin
        @test (@p maximum [1 2 3; 4 5 0] (x->x[2]))  ==  [2 5]'
        @test (@p minimum [1 2 3; 4 5 0] (x->x[2]))  ==  [3 0]'
    end
    @shouldtestset2 "valuemap" begin
        @test valuemap([0 1 2 3], [10 20 30])  ==  [0. 10. 20. 30.]
    end
    @shouldtestset2 "clamp" begin
        @test clamp(1,1,1)  ==  1
        @test clamp(0,1,2)  ==  1
        @test clamp(2,1,2)  ==  2
        @test clamp([-1 2 3 10; -5 10 11  100],[1 10],[3 15])  ==  [1 2 3 3; 10 10 11 15]
        @test clamp(Float32[-1 2 3 10; -5 10 11  100],[1 10],[3 15])  ==  [1 2 3 3; 10 10 11 15]
        @test clamp([1,2],[3,4],[5,6])   ==  [3,4]
        @test clamp([0 1 11; -5 3 20], rand(10,15))  ==  [1 1 10; 1 3 15]
        @test clamp([0 1 11; -5 3 20; -4 3 5], rand(10,15,4))  ==  [1 1 10; 1 3 15; 1 3 4]
    end
    @shouldtestset2 "rand" begin
        @test size(randsiz(siz(rand(2,3)))) == (2,3)
        @test eltype(randsiz([2 3]', Float32)) == Float32
    end
    @shouldtestset2 "plus" begin
        @test (@p plus zeros(2,3) ones(2,1))  ==  ones(2,3)
        @test (@p minus zeros(2,3) ones(2,1))  ==  -ones(2,3)
        @test (@p times ones(2,3) zeros(2,1))  ==  zeros(2,3)
        @test (@p divby ones(2,3) 2*ones(2,1))  ==  0.5*ones(2,3)
    end
end

@shouldtestset "utils" begin
    @shouldtestset2 "histdict" begin
        d = [Dict(:a => 1), Dict(:a => 1), Dict(:a => 10)]
        r = Dict(1 => 2, 10 => 1)
        @test histdict(d, :a)  ==  r
        @test histdict([1,1,10])  ==  r
    end
end

println("done!\n")
