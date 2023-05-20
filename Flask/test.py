import re
result = []
with open('requirements_new.txt') as f:
    lines = f.readlines()
    for line in lines:
        clean_text = re.sub(r'==.*', '', line.strip())
        result.append(clean_text)
print(result)

f = open("requirements_new2.txt", "x")
for content in result:
    f.write(content+"\n")
f.close()
