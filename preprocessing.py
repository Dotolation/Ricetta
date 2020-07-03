import re
#Preprocessing
#Title extraction

bad = "０１２３４５６７８９?!,．(){}[]\'\";:『』【】〜～•/・＆､＋"
good = "0123456789？！、.（）（）「」’’；：「」「」~~.／.&、+"

bad = bad + "ＡＣＤＥＦＧＩＪＬＭＮＰＱＲＳＴＵＶＷＸＹＺａｂｄｅｈｉｋｌｎｏｐｒｓｔｕｖｗｘｙｚｇｃｍＯＫＨＢℊ"
bad = bad + "ｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ"
bad = bad + "①②③④⑤⑥⑦⑧⑨⑴⑵⑶⑸⑹⒈⒉⒊⒌❶❷❸❹❼❾➀➁➂➄➅"
good = good + "12345678912356123512347912356"
good = good +"ACDEFGIJLMNPQRSTUVWXYZabdehiklnoprstuvwxyzgcmOKHBg"
good = good + "ァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン"


print (bad)
print (good)

dic = {}

for i in range(0,len(bad)):
    b = bad[i]
    dic[b] = good[i]

print (dic)
    
fix = str.maketrans(dic)


def preprocessing(which):
    with open("raw/"+ which +".txt", 'r', encoding='UTF-8') as file:
        text = file.read()
        k = text.translate(fix)
        #k = re.sub("\t", " <t> ", k)
        #k = re.sub("\n", "<nl>", k)
        #k = re.sub("\r", "<rnl>", k)
        #k = re.sub("\r\n", "<rnnl>", k)

        k = re.sub(r"&lt；", "＜", k)
        k = re.sub(r"&gt；", "＞", k)
        k = re.sub(r"&#39；", "’", k)
        k = re.sub(r"&quot；", "”", k)
        k = re.sub(r"&amp；", "&", k)

        
        k = re.sub("[0-9]{7}", "", k)
        k = re.sub("http(.*)", "", k)
        k = re.sub(r"([0-9]{4}|[0-9]{2})[／.年][0-9]{1,2}[／.月][0-9]{1,2}(日)?", " <date> ", k)

        k = re.sub("--- Title ---", "<title> ", k)
        k = re.sub("--- Ingredients ---", "<ingr> ",k)
        k = re.sub("--- Step ---", "<step>",k)
        k = re.sub("--- End ---", "<end><spl>",k)
        k = re.sub('[" "" ""　"]+', " ", k)
        k = re.sub('\n ', '\n', k)

        k = re.sub("[☆〇✪⭐*]", "★", k)
        k = re.sub("[✽✾✿❀❁※←↑→↓↗⇐⇒⇔⇦⇧⇨]", "✿", k)
        k = re.sub("[✡✣✤✧✩✭✰✱✲✳✴＊]", "✨", k)
        k = re.sub("[❄❅❆❇❋✺✻✼]", "♥", k)
        k = re.sub("[♡❣❥❤]", "♥", k)
        k = re.sub("[◎○◍⚫◯⭕⚪◉◒◔◕◾□♢◇◼◆■▪▲△▶▷▼▽▿◀]", "●", k)
        k = re.sub("[♬♫♩]", "♪", k)
        k = re.sub("[‼❕❗]", "！", k)
        k = re.sub("[⁈❓⁉]", "？", k)
        k = re.sub("[′‵`゛‘“”‟ﾞ]", "’", k)
        k = re.sub("[…┅]", "...", k)

        
        
        k = re.sub("[㎝㌢]", "cm", k)
        k = re.sub("㏄", "cc", k)
        k = re.sub("㎏", "kg", k)
        k = re.sub("㎜", "mm", k)
        k = re.sub("[㎖㍉]", "ml", k)
        k = re.sub("㍑", "l", k)
        k = re.sub("㌍", "カロリー", k)
        k = re.sub("㌘", "g", k)
        k = re.sub("㍗", "ワット", k)
        


        clin = re.sub("([^。？！、.（）｛｝「」’「」\s0-9A-Za-zぁ-ゔァ-ヴー一-龠々〆〤<>●★♪♥℃⁉✨✿~／%&+]|" ")", "", k)

        
        titles = open("cleaned/"+ which + ".txt", 'w', encoding='UTF-8')
        titles.write(clin)
        titles.close()
        

        lol = open("cleaned/"+ "rubishmojis" + ".txt", 'w', encoding='UTF-8')
        rub = re.sub("[。？！、.（）｛｝「」’「」\s0-9A-Za-zぁ-ゔァ-ヴー一-龠々〆〤<>●★♪♥℃✨✿~／%&+]", "", text)
        lol.write(''.join(sorted(set(rub))))
        lol.close()



def title_body_split(which):
    with open("segmented/"+ which +"_seg.txt", 'r', encoding='UTF-8') as file:
        k = file.read()
        k = re.sub("\t", " <t> ", k)
        k = re.sub("\n", "<nl>", k)
        k = re.sub('[" "" ""　"]+', " ", k)


        k = re.sub("< ", "<", k)
        k = re.sub(" >", ">",k) 
        k = re.sub(" <nl><ingr>", " </title>\n<ingr>",k)
        k = re.sub("<title> <nl>", "<title> ",k)

        body = open(which + "_BD.txt",'w', encoding='UTF-8')
        
        b = re.sub("<title>(.|' ')*</title>\n", "", k)
        b = re.sub("<end>", "<end> ", b)
        b = re.sub("<nl>", " \n", b)
        b = re.sub("<t>", " ", b)
        b = re.sub('[" "" ""　"]+', " ", b)
        
        body.write(b)
        body.close()


        titles = open(which + "_TI.txt", 'w', encoding='UTF-8')
         
        t = re.sub("<ingr>(.|" ")*<spl>", "",k)
        t = re.sub("</title>\n", "</title><spl> \n",t)
        t = re.sub("<nl>", "",t)
        t = re.sub("\n\n", "\n",t)
        t = re.sub('[" "" ""　"]+', " ", t)
        
        titles.write(t)
        titles.close()
        

      

preprocessing("test")
preprocessing("train")

title_body_split("test")
title_body_split("train")
        
