import os, time, json
from multiprocessing import Pool


def convert_one_design(design, output_dir):
    design_dir = f"../{cmd}/{design}.v"
    print('Current Design: ', design)
    print('Current CMD: ', cmd)
    os.system(f'python3 analyze.py {design_dir} -N {design} -C {cmd} -O {output_dir}')

    

if __name__ == '__main__':
    # cmd = 'ast' ## for word-level
    # cmd = 'sog' ## for bit-level
    global cmd
    # cmd = "ori"
    cmd = "pos"

    design_lst = ["spi", 
              "b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "b09", "b10",
              "b11", "b12", "b13", "b14", "b15", "b17", "b18", "b19", "b20",
              "b21", "b22",
              "fpu", "i2c_master_top", "mc_top", "pcm_slv_top" , "sasc_top", "simple_spi_top", "tv80s", "usb_phy", "usbf_top", "wb_dma_top"
              ]
    for design in design_lst:
        output_dir = f'../rtl_graph/{cmd}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        convert_one_design(design, output_dir)


    
    

    



