// Readers for the golden-model files (model/gen_layer_vectors.py,
// model/gen_desc_vectors.py, model/quantize.py program.txt). The UVM
// environment has no model of its own: all expected data comes from these.

typedef byte unsigned npu_bytes_t[$];

typedef struct {
  bit [31:0] w[10];
  int        err_code;
} npu_desc_vector_t;

// One layer case: descriptor fields in file order, input images and expected output.
class npu_layer_case extends uvm_object;
  `uvm_object_utils(npu_layer_case)

  int op, k, pad, in_h, in_w, c_in, c_out, in_base, out_base, w_base, b_base, rq_mult, rq_shift;
  int z_in, z_out, act_min, act_max;
  npu_bytes_t in_b, w_b, b_b, out_b;

  function new(string name = "npu_layer_case");
    super.new(name);
  endfunction

  function void pack(output bit [31:0] w[10]);
    w[0] = 32'(op & 'hF) | 32'(k & 'hF) << 4 | 32'(pad & 'hF) << 8;
    w[1] = 32'(in_h & 'hFFFF) | 32'(in_w & 'hFFFF) << 16;
    w[2] = 32'(c_in & 'hFFFF) | 32'(c_out & 'hFFFF) << 16;
    w[3] = 32'(in_base);
    w[4] = 32'(out_base);
    w[5] = 32'(w_base);
    w[6] = 32'(b_base);
    w[7] = 32'(rq_mult);
    w[8] = 32'(rq_shift & 'h3F);
    w[9] = 32'(z_in & 'hFF) | 32'(z_out & 'hFF) << 8 | 32'(act_min & 'hFF) << 16 | 32'(act_max & 'hFF) << 24;
  endfunction

  function string convert2string();
    return $sformatf("op %0d k %0d pad %0d %0dx%0d c %0d->%0d in %h out %h", op, k, pad, in_h, in_w, c_in, c_out,
                     in_base, out_base);
  endfunction
endclass

class npu_image extends uvm_object;
  `uvm_object_utils(npu_image)
  int         idx, label, expected, in_base;
  npu_bytes_t pixels;
  int         trace_layer[$], trace_base[$];
  npu_bytes_t trace_bytes[$];

  function new(string name = "npu_image");
    super.new(name);
  endfunction
endclass

class npu_program extends uvm_object;
  `uvm_object_utils(npu_program)
  int         n_layers;
  bit [31:0]  desc[16][10];
  bit [31:0]  weights[$];
  npu_image   images[$];

  function new(string name = "npu_program");
    super.new(name);
  endfunction
endclass

class npu_vectors;
  static function int open(string path);
    int fd = $fopen(path, "r");
    if (fd == 0) `uvm_fatal("VEC/OPEN", {"cannot open ", path})
    return fd;
  endfunction

  static function void read_bytes(int fd, int n, ref npu_bytes_t q);
    int v;
    q.delete();
    for (int i = 0; i < n; i++) begin
      if ($fscanf(fd, "%d", v) != 1) `uvm_fatal("VEC/FMT", "short byte section")
      q.push_back(8'(v));
    end
  endfunction

  static function void hex_bytes(string hex, ref npu_bytes_t q);
    q.delete();
    for (int i = 0; i + 1 < hex.len(); i += 2) begin
      string h = hex.substr(i, i + 1);
      q.push_back(8'(h.atohex()));
    end
  endfunction

  // gen_layer_vectors.py: CASE line, then IN / W / B / OUT sections.
  static function void read_layer_cases(string path, ref npu_layer_case q[$]);
    int fd = open(path);
    string tag;
    int n;
    npu_layer_case c;
    while ($fscanf(fd, "%s", tag) == 1) begin
      case (tag)
        "CASE": begin
          c = npu_layer_case::type_id::create($sformatf("case_%0d", q.size()));
          if ($fscanf(fd, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
                      c.op, c.k, c.pad, c.in_h, c.in_w, c.c_in, c.c_out, c.in_base, c.out_base, c.w_base, c.b_base,
                      c.rq_mult, c.rq_shift, c.z_in, c.z_out, c.act_min, c.act_max) != 17)
            `uvm_fatal("VEC/FMT", "short CASE line")
          q.push_back(c);
        end
        "IN", "W", "B", "OUT": begin
          if (c == null || $fscanf(fd, "%d", n) != 1) `uvm_fatal("VEC/FMT", {"misplaced ", tag})
          case (tag)
            "IN":    read_bytes(fd, n, c.in_b);
            "W":     read_bytes(fd, n, c.w_b);
            "B":     read_bytes(fd, n, c.b_b);
            default: read_bytes(fd, n, c.out_b);
          endcase
        end
        default: `uvm_fatal("VEC/FMT", {"unknown tag ", tag})
      endcase
    end
    $fclose(fd);
  endfunction

  // gen_desc_vectors.py: w0 .. w9 err_code per line.
  static function void read_desc_vectors(string path, ref npu_desc_vector_t q[$]);
    int fd = open(path);
    npu_desc_vector_t v;
    forever begin
      if ($fscanf(fd, "%d %d %d %d %d %d %d %d %d %d %d", v.w[0], v.w[1], v.w[2], v.w[3], v.w[4], v.w[5], v.w[6],
                  v.w[7], v.w[8], v.w[9], v.err_code) != 11) break;
      q.push_back(v);
    end
    $fclose(fd);
  endfunction

  // quantize.py program.txt: LAYERS, DESC, WORDS, then IMAGE lines each
  // followed by its TRACE lines (layer outputs), if any.
  static function npu_program read_program(string path);
    npu_program p = npu_program::type_id::create("program");
    int fd = open(path);
    string tag, hex;
    int idx, n, img, layer, base, label, expected;
    bit [31:0] w;
    npu_image im;
    while ($fscanf(fd, "%s", tag) == 1) begin
      case (tag)
        "LAYERS": void'($fscanf(fd, "%d", p.n_layers));
        "DESC": begin
          void'($fscanf(fd, "%d", idx));
          for (int k = 0; k < 10; k++) begin
            void'($fscanf(fd, "%d", w));
            p.desc[idx][k] = w;
          end
        end
        "WORDS": begin
          void'($fscanf(fd, "%d", n));
          for (int i = 0; i < n; i++) begin
            void'($fscanf(fd, "%h", w));
            p.weights.push_back(w);
          end
        end
        "IMAGE": begin
          im = npu_image::type_id::create("image");
          void'($fscanf(fd, "%d %d %d %d %d %s", im.idx, im.label, im.expected, im.in_base, n, hex));
          hex_bytes(hex, im.pixels);
          if (im.pixels.size() != n) `uvm_fatal("VEC/FMT", $sformatf("image %0d: %0d bytes, header says %0d",
                                                im.idx, im.pixels.size(), n))
          p.images.push_back(im);
        end
        "TRACE": begin
          npu_bytes_t b;
          void'($fscanf(fd, "%d %d %d %d %s", img, layer, base, n, hex));
          hex_bytes(hex, b);
          if (im == null || img != im.idx || b.size() != n) `uvm_fatal("VEC/FMT", "misplaced TRACE")
          im.trace_layer.push_back(layer);
          im.trace_base.push_back(base);
          im.trace_bytes.push_back(b);
        end
        default: `uvm_fatal("VEC/FMT", {"unknown tag ", tag})
      endcase
    end
    $fclose(fd);
    return p;
  endfunction
endclass
