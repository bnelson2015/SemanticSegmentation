function img = matReader(filename)
    inp = load(filename);
    f = fields(inp);
    img = inp.(f{1});
    img = img(:,:,1:end-1);
end

