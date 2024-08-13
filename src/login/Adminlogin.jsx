import React, { useState } from "react";
import "./Login.css";
import { Link, useNavigate } from "react-router-dom";
import { Flex } from "@chakra-ui/layout";
import { Image } from "@chakra-ui/image";
import Navbar from "../UserSide/Components/Home/Navbar";
import Footer from "../UserSide/Components/Home/Footer";

// import admin from "./assets/admin.jpg";

const AdminLogin = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();
  const validateForm = () => {
    let isValid = true;
    if (!email || !password) {
      setError("Please fill all the fields");
      isValid = false;
    } else if (!/\S+@\S+\.\S+/.test(email)) {
      setError("Invalid email address");
      isValid = false;
    } else if (password.length < 8) {
      setError("Password must be at least 8 characters long");
      isValid = false;
    }
    return isValid;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    if (validateForm()) {
      try {
        // await login(email, password);
        localStorage.setItem("adminEmail", JSON.stringify(email));
        navigate("/dashboard");
      } catch (err) {
        setError(err);
      }
    }
  };
  return (
    <>
      <Navbar />
      <div className="login">
        <Flex align="center" className="loginContainer">
          <Flex
            className="logoanime "
            ml={"2rem"}
            justify={"center"}
            align="center"
            height={"250px"}
            overflow={"hidden"}
            p="2rem"
            display={{ base: "none", md: "Flex" }}
          >
            <Image
              src={"https://imgs.search.brave.com/TpezS77NJwwmump0_RK1zN0KWprPLt6oRq-kOqqNbPE/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly9jZG4u/aWNvbnNjb3V0LmNv/bS9pY29uL2ZyZWUv/cG5nLTI1Ni9mcmVl/LW15bnRyYS0yNzA5/MTY4LTIyNDkxNTgu/cG5nP2Y9d2VicCZ3/PTEyOA"}
              alt="myntra2"
            />
          </Flex>
          <div className="loginDetail">
            <div>
              <h3>Admin Login</h3>
            </div>

            <div>
              <form onSubmit={handleSubmit}>
                <input
                  name="email"
                  onChange={(e) => setEmail(e.target.value)}
                  type="email"
                  placeholder="Enter email"
                  className={
                    error && (!email || !/\S+@\S+\.\S+/.test(email))
                      ? "invalid"
                      : ""
                  }
                />
                {error && (!email || !/\S+@\S+\.\S+/.test(email)) && (
                  <span className="error">
                    {"* "}
                    {error}
                  </span>
                )}
                <input
                  name="password"
                  onChange={(e) => setPassword(e.target.value)}
                  type="password"
                  placeholder="Enter a password"
                  className={error && password.length < 8 ? "invalid" : ""}
                />
                {error && password.length < 8 && (
                  <span className="error">
                    {"* "}
                    {error}
                  </span>
                )}
                <p>
                  New User ? <Link to="/signup">Signup </Link>
                </p>

                <button type="submit">Login</button>
              </form>
            </div>
          </div>
        </Flex>
      </div>
      <Footer />
    </>
  );
};

export default AdminLogin;
