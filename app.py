import os
import json
import numpy as np
import scipy as sp
import scipy.io.wavfile
import gc, re, string
from matplotlib import pyplot as plt
import IPython.display as ipd
import pandas as pd
import soundfile as sf

from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)


def print_plot_play(lenght, data, text=''):
 
    time = np.linspace(0., data.shape[0], data.shape[0])
    plt.plot(time, data[:, 0], label="Left channel")
    plt.plot(time, data[:, 1], label="Right channel")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()
#trường mở rộng tên file
ALLOWED_EXTENSIONS = set(['wav'])
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode(fname, msg):
    # tiến hành đọc file .wav: trả về 1 sample rate (sample/s) và data từ file wav
    # ở đây sample rate là 1 int, data là 1 numpy array 
    # giả sử channels.shape = 14595, rate = 22050
    #1: chia tín hiệu gốc thành n khối (segments) liên tiếp.
    rate, channels = sp.io.wavfile.read(fname,True) #hàm sp.io.wavfile.read return rate : int, data : numpy array(channels)
    print('rate = ',rate)
    print('channels = ',channels) #channels là mảng np 2 chiều (length,2)
    print('channels length = ',channels.shape[0])
    print('channels dtype = ',channels.dtype)
    channels = channels.copy() #shallow copy (copy đ/c vùng nhớ): vẫn còn tham chiếu lên đối tượng cũ  
    channels_float32 = channels / 32768.0 # Convert to [-1.0, +1.0]
    print('channels dtype = ',channels_float32) # in ra màn hình cái biểu đồ biên độ xem 
    print_plot_play(channels_float32,channels_float32,"Vinh") 
    # print_plot_play(rate,channels.reval(),"")
    # độ dài messege, giả sử msglen = 168
    # vì một ký tự đổi ra mã ascii phải cần 8 bit để lưu mã ascii đó
    msglen = 8 * len(msg) # tại vì 1 kí tự ascii là 1 byte = 8 bit nên * 8

    # xử lý data của file dựa trên dữ liệu messege
    # tính độ dài 1 segment, và số segment
    # seglen = 2  * 2^(log2(2*msglen)) = 2 * 2^(log2(2*168)) = 1024 ( log2(2*168) : được làm tròn lên )
    # segnum =  14595/1024 = 15 
    # .shape là thuộc tính của mảng numpy có kdl tuple lưu thông tin về các chiều dài của mảng theo mỗi chiều
    seglen = int(2 * 2**np.ceil(np.log2(2*msglen))) #hàm ceil là hàm làm tròn lên
    print('seglen = ',seglen)
    segnum = int(np.ceil(channels.shape[0]/seglen)) # lấy số hàng của ma trận / seglen. vì có thể channels là ma trận 2 hoặc nhiều chiều.
    print('segnum = ',segnum)

    # nếu mảng chỉ là mảng 1 chiều
    if len(channels.shape) == 1: # nếu channels là mảng 1 chiều thì biến thành mảng 2 chiều
        # thay đổi hình dạng và kích thước của mảng tại chỗ
        # .resize(new_shape, refcheck=True)
        # If False, reference count will not be checked. Default is True
        # refcheck = false thì có thể thay đổi kích thước của mảng tham chiếu
        channels.resize(segnum*seglen, refcheck=False)
        # newaxis được sử dụng để tăng kích thước của mảng hiện có thêm một chiều , khi được sử dụng một lần...
        # ...Mảng 1D sẽ trở thành mảng 2D
        channels = channels[np.newaxis]
    else:
        channels.resize((segnum*seglen, channels.shape[1]), refcheck=False)
        # chuyển vị ma trận của channels
        channels = channels.T
        print('chuyen vi channels = ',np.count_nonzero(channels==0))

    #4: mã hóa dữ liệu nhị phân messege 
    # thực chất là tạo phase new dựa trên bit 0 1
    # numpy.ravel(array, order = ‘C’) : trả về 1 mảng tiếp giáp phẳng (mảng 1D vs tất cả các phần tử mảng đầu vào,...
    # ... tức mảng 1 chiều)
    msgbin = np.ravel([[int(y) for y in format(ord(x), '08b')] for x in msg]) #vòng lặp này sẽ biến đổi các ký tự của msg thành dạng nhị phân
    print('msgbin =',msgbin)
    msgPi = msgbin.copy() 
    
    msgPi[msgPi == 0] = -1 # nếu =0 thì gán =-1 để nhân với pi/2
    msgPi = msgPi * -np.pi/2 # nếu là bit 1 thì = -pi/2
    print('msgPi =',msgPi)
    # 2: 
    # trả về 1 shape mới cho 1 mảng mà ko thay đổi dữ liệu của nó
    # để thực hiện fourier rời rạc dft
    segs = channels[0].reshape((segnum,seglen)) # chia channels thành n đoạn segment
    print('segs =',segs)
    print('segs lenght =',segs[0])
    # tính toán biến đổi fourier rời rạc một chiều DFT
    segs = np.fft.fft(segs)
    print('segs sau khi FFT =',segs)
    # lấy trị tuyệt đối của segs => cường độ magnitude[i]
    M = np.abs(segs) # lấy cường độ A
    print('M  =',M )
    # trả về góc của segs => phase[i]
    P = np.angle(segs) # lấy pha phi
    print('P  =',P )
    #3:
    #tính toán sự chênh lệch giữa các phase
    dP = np.diff(P, axis=0) # tính độ lệch pha theo trục dọc nên dP.length = P.length-1
    # Ví dụ:
    # [[0 ,2] 
    # ,[2,3]
    # ,[3,4]]
    # thì độ lệch dP là 
    # [[2 1]
    # ,[1 1]]
    print('dP  =',dP )
    #5,6: và các bc để điều chỉnh pha của đoạn đầu
    #chia làm tròn xuống: L/2
    segmid = seglen // 2
    print("segmid = ",segmid)
    P[0,-msglen+segmid:segmid] = msgPi #Chỉ lấy segment đầu tiên giấu tin
    #chỗ này là từ giữa trở về trước đoạn = msglen 

    # trích xuất những phần tử = -1 trong mảng msgPi
    print("P[0] = ", P[0])
    P[0,segmid+1:segmid+1+msglen] = -msgPi[::-1]   #chỗ này là từ giữa trở về sau đoạn = msglen
    #đoạn này bằng nghịch đảo của -msgPi để đối xứng khi biến đổi Fourier
    print("P trich xuat -1 = ", P[0])
    for i in range(1, len(P)): P[i] = P[i-1] + dP[i-1] # chỗ này không ảnh hưởng tới P[0](chỗ giấu tin)
    print("P cong voi dP = ", P)
    #7: 
    # Tính toán số mũ của tất cả các phần tử trong mảng đầu vào. Trả về 1 mảng mới vs hàm theo mũ của từng phần tử
    segs = (M * np.exp(1j * P))
    print("segs = (M * np.exp(1j * P) = ", segs)
    #ifft: fft nghịch đảo để tái tạo tín hiệu cho từng khối
    segs = np.fft.ifft(segs).real
    print("np.fft.ifft(segs).real = ", segs)
    channels[0] = segs.ravel().astype(np.int16) #.ravel biến mảng nhiều chiều thành mảng 1 chiều
    print("channels[0] =",channels[0])
    sp.io.wavfile.write('file_da_giau_tin_'+fname, rate, channels.T)
    return 'file_da_giau_tin_'+fname

def is_alpha(word):
    try:
        return word.encode('ascii').isalpha()
    except:
        return False

def decode(fname, msgLength):
    resultList = []
    # chia audio đã mã hóa thành nhiều khối (segments) liến tiếp
    rate, channels = sp.io.wavfile.read(fname)
    rate, channels.shape
    print('rate = ',rate)
    print('channels = ',channels) #channels là mảng np 2 chiều (length,2)
    print('channels length = ',channels.shape[0])
    # cho biến flag = 0
    flag = 0
    # nếu flag khác 0 thì kết thức quá trình decode -> result[] 
    while flag == 0:
        # chiều dài của messege (theo kí tự)
        y = int(msgLength)
        # chạy vòng lặp for từ 3 -> y để giải mã từng kí tự
        for i in range(3,y):
            msglen = 8*i
            seglen = 2*int(2**np.ceil(np.log2(2*msglen)))
            # L / 2 
            segmid = seglen // 2

            if len(channels.shape) == 1:
                x = channels[:seglen] #lấy seglen phần tử đầu tiên
            else:
                x = channels[:seglen,0] #lấy seglen phần tử đầu tiên theo chiều ngang (trục hoành)

            # giải mã
            x = (np.angle(np.fft.fft(x))[segmid-msglen:segmid] < 0).astype(np.int8)
            print('x = ', x)
            # dot nhận vào 2 mảng và trả về 1 mảng đặc biệt được tạo thành từ những yêu cầu khác nhau
            # arange(start, end, step) trả về mảng các giá trị cách đều nhau, vs điều kiện là tham số truyền vào
            x = x.reshape((-1,8)).dot(1 << np.arange(8 - 1, -1, -1))
            print('x.reshape((-1,8)).dot(1 << np.arange(8 - 1, -1, -1)) = ', x)
            #formating lại x theo định dạng unicode đầu vào
            # formating dưới dạng kí tự char, trả về mảng mới theo định dạng đó
            text = ''.join(np.char.mod('%c',x)) 


            # kiểm tra text[0] có nằm trong bảng chữ cái alpha, nếu đúng thì ...
            if is_alpha(text[0]):
                # ...dùng biểu thức chính quy re.sub() để tìm những ký tự thích hợp
                text = re.sub(r'[^a-zA-Z0-9,._ ]', r'', text)
                # thêm phần tử vào cuối mảng kết quả
                resultList.append(text)
            else:
                # nếu ko thì giải phóng bộ nhớ
                gc.collect()

        break    
    return resultList   


#chạy web

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/showEncode')
def showEncode():
   return render_template('encode.html')

@app.route('/showDecode')
def showDecode():
   return render_template('decode.html')

@app.route('/handleEncode', methods = ['POST'])  
def handleEncode():  
    if request.method == 'POST':  
        audioFile = request.files['file']  
        secretMsg = request.form['secret']
        newAudioName = 'encodingAudio.wav'
        audioFile.save(newAudioName)
        if os.path.exists('file_da_giau_tin_encodingAudio.wav'):
            os.remove('file_da_giau_tin_encodingAudio.wav') 
        encode(newAudioName, secretMsg) 
        os.remove(newAudioName)
        return jsonify(
            error=False,
            data='file_da_giau_tin_encodingAudio.wav'
        )

@app.route('/handleDecode', methods = ['POST'])  
def handleDecode():  
    if request.method == 'POST':  
        audioFile = request.files['file']  
        secretLength = request.form['secret']
        
        newAudioName = 'decodingAudio.wav'
        if os.path.exists(newAudioName):
            os.remove(newAudioName) 

        audioFile.save(newAudioName)
        data = decode(newAudioName, secretLength)
        os.remove(newAudioName)
        return jsonify(
            error=False,
            data=data
        )


@app.route('/downloadFile/<filename>')
def downloadFile(filename):
    file_path = filename
    return send_file(file_path, as_attachment=True, attachment_filename='')

if __name__ == '__main__':
  app.run()


#tại sao phải biến đổi forier? tại vì channel bình thường là theo thời gian nên cần phải biến đổi forier để channel theo tần số


#Thuận toán: file audio - > lấy data audio -> do nó có 2 channel nên chỉ lấy channel đầu đi encode (phần này thì lên tìm hiểu về numpy của python trước) 
# -> chuyển vị ma trận ban đầu (cái này in ra màn hình rồi ấy check để hiểu thêm) - > tính các thông số segment 
# -> tách thành các segment với độ dài N - > chỉ lấy đoạn đầu P[0] để mã hóa các đoạn khác k quan tâm 
# -> lấy biên độ M, lấy góc angel P và độ lệch pha dP (những cái này để lúc Fourier ngược lại dùng ->bước này thật ra k có cũng được) 
# -> gán mảng msgPi(mảng này là gán 0 1 Pi/2) vào đoạn giữa của angel P (chỗ này gán ở đâu trên P cũng được nhưng khi encode phải biết vị trí) 
# -> sau đó lấy góc P cộng với biên độ với độ lệch dP (chỗ này t cũng chưa rõ ) nhưng góc của nó lại đúng âm dương với mảng msgPi
# -> sau đó Fourier ngược lại ifft (test biến đổi fft thì vẫn ra đúng mảng đã biến đổi)

# decode thì chỉ cần biến đổi fft đoạn giữa rồi lấy ra mảng 0 1 rồi biến qua ASCII là được thông tin
