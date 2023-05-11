const body = document.querySelector("body")

const wiper = document.createElement("div")
wiper.classList.add("wiper")

const wiperImg = document.createElement("img")
wiperImg.setAttribute("src", "../static/logo_white.svg")

const wiperTxtHolder = document.createElement("div")
const wiperTxt = document.createElement("h2")
wiperTxt.innerHTML = "wanderers"

wiperTxtHolder.appendChild(wiperTxt)

wiper.appendChild(wiperImg)
wiper.appendChild(wiperTxtHolder)
body.appendChild(wiper)

barba.use(barbaPrefetch)


barba.init({
    debug: true,
    transitions: [
        {
            name: "next",
            custom({ current, next, trigger }){
                return (trigger.classList && trigger.classList.contains("next")) || trigger === "forward"
            },
            leave({ current, next, trigger }){
                return new Promise((resolve) => {
                    const timeline = gsap.timeline({
                        onComplete(){
                            current.container.remove()
                            resolve()
                        }
                    })

                    const navigation = current.container.querySelectorAll("header, a.next, a.previous")
                    const photos = current.container.querySelectorAll("div.photos")

                    timeline
                        .set(wiper, { x: "-100%" })
                        .set(wiperImg, { opacity: 0 })
                        .set(wiperTxt, { y: "100%" })
                        .to(navigation, { opacity: 0 }, 0)
                        .to(photos, { opacity: 0.25, x: 500 }, 0)
                        .to(wiper, { x: "0"}, 0)
                })
            },

            beforeEnter({ current, next, trigger }){
                return new Promise((resolve) => {
                    const timeline = gsap.timeline({
                        defaults: {
                            duration: 1
                        },
                        onComplete(){
                            resolve()
                        }
                    })

                    wiperTxt.innerHTML = next.container.getAttribute("data-title")

                    timeline
                        .to(wiperImg, { opacity: 1 }, 0)
                        .to(wiperTxt, { y: "0" }, 0)
                        .to(wiperTxt, { y: "100%" }, 2)
                        .to(wiperImg, { opacity: 0 }, 2)

                    window.scrollTo({
                        top: 0,
                        behavior: "smooth",
                    })
                })
            },
            
            enter({ current, next, trigger }){
                return new Promise((resolve) => {
                    const timeline = gsap.timeline({
                        onComplete(){
                            resolve()
                        }
                    })

                    const navigation = next.container.querySelectorAll("header, a.next, a.previous")
                    const photos = next.container.querySelectorAll("div.photos")

                    timeline
                        .set(navigation, { opacity: 0 })
                        .set(photos, { opacity: 0.25, x: -500 })
                        .to(navigation, { opacity: 1 }, 0)
                        .to(photos, { opacity: 1, x: 0 }, 0)
                        .to(wiper, { x: "100%" }, 0)
                })
            }
        },
        {
            name: "previous",
            leave({ current, next, trigger }){
                return new Promise((resolve) => {
                    const timeline = gsap.timeline({
                        onComplete(){
                            current.container.remove()
                            resolve()
                        }
                    })

                    const navigation = current.container.querySelectorAll("header, a.next, a.previous")
                    const photos = current.container.querySelectorAll("div.photos")

                    timeline
                        .set(wiper, { x: "100%" })
                        .set(wiperImg, { opacity: 0 })
                        .set(wiperTxt, { y: "100%" })
                        .to(navigation, { opacity: 0 }, 0)
                        .to(photos, { opacity: 0.25, x: -500 }, 0)
                        .to(wiper, { x: "0"}, 0)
                })
            },

            beforeEnter({ current, next, trigger }){
                return new Promise((resolve) => {
                    const timeline = gsap.timeline({
                        defaults: {
                            duration: 1
                        },
                        onComplete(){
                            resolve()
                        }
                    })

                    wiperTxt.innerHTML = next.container.getAttribute("data-title")

                    timeline
                        .to(wiperImg, { opacity: 1 }, 0)
                        .to(wiperTxt, { y: "0" }, 0)
                        .to(wiperTxt, { y: "100%" }, 2)
                        .to(wiperImg, { opacity: 0 }, 2)

                    window.scrollTo({
                        top: 0,
                        behavior: "smooth",
                    })
                })
            },
            
            enter({ current, next, trigger }){
                return new Promise((resolve) => {
                    const timeline = gsap.timeline({
                        onComplete(){
                            resolve()
                        }
                    })

                    const navigation = next.container.querySelectorAll("header, a.next, a.previous")
                    const photos = next.container.querySelectorAll("div.photos")

                    timeline
                        .set(navigation, { opacity: 0 })
                        .set(photos, { opacity: 0.25, x: 500 })
                        .to(navigation, { opacity: 1 }, 0)
                        .to(photos, { opacity: 1, x: 0 }, 0)
                        .to(wiper, { x: "-100%" }, 0)
                })
            }
        }
    ],
    views: []
})
