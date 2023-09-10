import pandas as pd 


def find(acc_number):
    x=df['Account Number'].index
    sum=0
    r=0
    for i in x:
        if(df['Account Number'][i]==acc_number):
            r=1
        else:
            sum+=1 
    if sum==len(x):
        r=-1
    return r
    
file_path="SBIAccountHolder.csv" 
df=pd.read_csv(file_path) 

print("Account Holder Details\n")
print(df)

choice=0
while choice!=6:
    print("\nAccount Options\n")
    print("1.Append Record\n")
    print("2.Delete Record\n")
    print("3.Credit amount\n")
    print("4.Debit amount\n") 
    print("5.Account Details\n")
    choice=int(input("Enter the choice : "))
    
    if choice==1:
        print("Adding Record\n")
        s=df.columns
        lis=[]
        for i in range(len(s)):
            str1="Enter the {} of new Holder\n".format(s[i])
            if s[i]=='Account Number' or s[i]=='Balance':
                x=int(input(str1))
            else:
                x=input(str1)
            lis.append(x)
        lis1=[]
        lis1.append(lis)
        df1=pd.DataFrame(lis1,columns=['Name', 'Account Number', 'Account Type', 'Adhaar_No', 'Balance'])
        df=df.append(df1,ignore_index=True)
        print("\n Added Succesfully")
    elif choice==2:
        print("Deleting Record\n")
        acc_number=int(input("enter the account number\n"))
        x1=df.shape[0]
        condition = df['Account Number'] == acc_number
        df = df.drop(index=df[condition].index)
        x2=df.shape[0]
        if x1==x2:
            print("Enter a valid account number\n")
        else:
            print("\n Deleted Succesfully")
    elif choice==3:
        acc_number=int(input("enter the account number\n"))
        fi=find(acc_number)
        if fi>0:
            condition = df['Account Number'] == acc_number
            i1=df[condition].index
            credit_amount=int(input("enter the amount to be credited\n"))
            tr=df['Balance'][i1]
            tr+=credit_amount
            df['Balance'][i1]=tr
            print("Amount Credited Successfully\n")
        else:
            print("Type a valid Account Number\n")
    elif choice==4:
        acc_number=int(input("enter the account number\n"))
        fi=find(acc_number)
        if fi>0:
            condition = df['Account Number'] == acc_number
            i1=df[condition].index
            debit_amount=int(input("enter the amount to be debited\n"))
            tr=df['Balance'][i1]
            tr=int(tr)
            tr-=debit_amount
            if tr<0:
                print("Error in debiting \n")
            else:
                df['Balance'][i1]=tr
                print("Amount Debited Succesfully\n")
        else:
            print("Type a valid Account Number\n")
    elif choice ==5:
        acc_number=int(input("enter the account number\n"))
        fi=find(acc_number)
        if fi>0:
            condition = df['Account Number'] == acc_number
            i1=df[condition].index
            print(df.iloc[i1])
        else:
            print("Type a valid Account Number\n")
    elif choice==6:
        print('Exit\n')
        break
    else:
        print("Not valid Input\n")
    print(df)
    df.to_csv('SBIAccountHolder.csv', index=False)