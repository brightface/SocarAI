import 'package:flutter/material.dart';
import 'package:flutter_signin_button/flutter_signin_button.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:mdw_app/tab_page.dart';

class LoginPage extends StatelessWidget {
  final GoogleSignIn _googleSignIn = GoogleSignIn();
  final FirebaseAuth _auth = FirebaseAuth.instance;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: _buildAppBar(),
      body: _buildBody(context),
    );
  }


  Widget _buildAppBar() {
    return AppBar(
      title: Text(
        'Log in',
        style: TextStyle(
            color:Colors.black,
            fontWeight: FontWeight.bold
        ),
      ),
    );
  }

  Widget _buildBody(BuildContext context){
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(
              '사용자를 등록합니다',
              style: TextStyle(fontSize: 32.0, fontWeight: FontWeight.bold),
            ),
            Padding(
              padding: EdgeInsets.all(50.0),
            ),
            SignInButton(
              Buttons.Google,
              onPressed: (){
                _handleSignIn().then((user){
                  Navigator.pushReplacement(context,
                      MaterialPageRoute(builder: (context) => TabPage(user)));
                }
               );
              },
           )
         ],
      ),
    );
  }

  Future<FirebaseUser> _handleSignIn() async{
    GoogleSignInAccount googleUser = await _googleSignIn.signIn();
    GoogleSignInAuthentication googleAuth = await googleUser.authentication;
    FirebaseUser user = await _auth.signInWithCredential(
      GoogleAuthProvider.getCredential(idToken: googleAuth.idToken,  accessToken: googleAuth.accessToken)
    );
    return user;
  }
}