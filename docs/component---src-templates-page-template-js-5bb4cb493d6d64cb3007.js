(window.webpackJsonp=window.webpackJsonp||[]).push([[11],{"/d1K":function(e,t,a){"use strict";a.d(t,"a",(function(){return I}));a("E5k/");var n=a("2rtX"),i=a("q1tI"),r=a.n(i),l=a("Wbzz"),c=(a("pJf4"),a("iSRb")),o=a.n(c),s=function(e){var t=e.author,a=e.isIndex;return r.a.createElement("div",{className:o.a.author},r.a.createElement(l.Link,{to:"/"},r.a.createElement("img",{src:Object(l.withPrefix)(t.photo),className:o.a.author__photo,width:"100",height:"115",alt:t.name})),a?r.a.createElement("h1",{className:o.a.author__title},r.a.createElement(l.Link,{className:o.a["author__title-link"],to:"/"},t.name)):r.a.createElement("h2",{className:o.a.author__title},r.a.createElement(l.Link,{className:o.a["author__title-link"],to:"/"},t.name)),r.a.createElement("p",{className:o.a.author__subtitle},t.bio))},m=(a("rzGZ"),a("Dq+y"),a("8npG"),a("Ggvi"),a("7Qib")),u=a("euHg"),_=a.n(u),d=function(e){var t=e.icon;return r.a.createElement("svg",{className:_.a.icon,viewBox:t.viewBox},r.a.createElement("path",{d:t.path}))},p=a("aU/I"),h=a.n(p),b=function(e){var t=e.contacts;return r.a.createElement("div",{className:h.a.contacts},r.a.createElement("ul",{className:h.a.contacts__list},Object.keys(t).map((function(e){return r.a.createElement("li",{className:h.a["contacts__list-item"],key:e},r.a.createElement("a",{className:h.a["contacts__list-item-link"],href:Object(m.a)(e,t[e]),rel:"noopener noreferrer",target:"_blank"},r.a.createElement(d,{icon:Object(m.b)(e)})))}))))},g=a("Nrk+"),E=a.n(g),f=function(e){var t=e.copyright;return r.a.createElement("div",{className:E.a.copyright},t)},k=a("je8k"),v=a.n(k),y=function(e){var t=e.menu;return r.a.createElement("nav",{className:v.a.menu},r.a.createElement("ul",{className:v.a.menu__list},t.map((function(e){return r.a.createElement("li",{className:v.a["menu__list-item"],key:e.path},r.a.createElement(l.Link,{to:e.path,className:v.a["menu__list-item-link"],activeClassName:v.a["menu__list-item-link--active"]},e.label))}))))},N=a("SySy"),x=a.n(N),w=function(e){var t=e.data,a=e.isIndex,n=t.site.siteMetadata,i=n.author,l=n.copyright,c=n.menu;return r.a.createElement("div",{className:x.a.sidebar},r.a.createElement("div",{className:x.a.sidebar__inner},r.a.createElement(s,{author:i,isIndex:a}),r.a.createElement(y,{menu:c}),r.a.createElement(b,{contacts:i.contacts}),r.a.createElement(f,{copyright:l})))},I=function(e){return r.a.createElement(l.StaticQuery,{query:"3187933791",render:function(t){return r.a.createElement(w,Object.assign({},e,{data:t}))},data:n})}},"2rtX":function(e){e.exports=JSON.parse('{"data":{"site":{"siteMetadata":{"title":"Tyler Kirby","subtitle":"Data scientist, classicist","copyright":"© All rights reserved.","menu":[{"label":"Articles","path":"/"},{"label":"About me","path":"/pages/about"},{"label":"Contact me","path":"/pages/contacts"}],"author":{"name":"Tyler Kirby","photo":"/profile.jpg","bio":"Data scientist, classicist","contacts":{"github":"https://github.com/TylerKirby","email":"mailto:tyler.kirby9398@gmail.com"}}}}}}')},"8vKr":function(e,t,a){"use strict";a.r(t),a.d(t,"query",(function(){return o}));var n=a("q1tI"),i=a.n(n),r=a("Zttt"),l=a("/d1K"),c=a("RXmK"),o="4092021430";t.default=function(e){var t=e.data,a=t.site.siteMetadata,n=a.title,o=a.subtitle,s=t.markdownRemark.frontmatter,m=s.title,u=s.description,_=t.markdownRemark.html,d=null!==u?u:o;return i.a.createElement(r.a,{title:m+" - "+n,description:d},i.a.createElement(l.a,null),i.a.createElement(c.a,{title:m},i.a.createElement("div",{dangerouslySetInnerHTML:{__html:_}})))}},"Nrk+":function(e,t,a){e.exports={copyright:"Copyright-module--copyright--1ariN"}},RBgx:function(e,t,a){e.exports={page:"Page-module--page--2nMky",page__inner:"Page-module--page__inner--2M_vz",page__title:"Page-module--page__title--GPD8L",page__body:"Page-module--page__body--Ic6i6"}},RXmK:function(e,t,a){"use strict";a.d(t,"a",(function(){return c}));var n=a("q1tI"),i=a.n(n),r=a("RBgx"),l=a.n(r),c=function(e){var t=e.title,a=e.children,r=Object(n.useRef)();return Object(n.useEffect)((function(){r.current.scrollIntoView()})),i.a.createElement("div",{ref:r,className:l.a.page},i.a.createElement("div",{className:l.a.page__inner},t&&i.a.createElement("h1",{className:l.a.page__title},t),i.a.createElement("div",{className:l.a.page__body},a)))}},SySy:function(e,t,a){e.exports={sidebar:"Sidebar-module--sidebar--X4z2p",sidebar__inner:"Sidebar-module--sidebar__inner--Jdc5s"}},"aU/I":function(e,t,a){e.exports={contacts:"Contacts-module--contacts--1rGd1",contacts__list:"Contacts-module--contacts__list--3OgdW","contacts__list-item":"Contacts-module--contacts__list-item--16p9q","contacts__list-item-link":"Contacts-module--contacts__list-item-link--2MIDn"}},euHg:function(e,t,a){e.exports={icon:"Icon-module--icon--Gpyvw"}},iSRb:function(e,t,a){e.exports={author__photo:"Author-module--author__photo--36xCH",author__title:"Author-module--author__title--2CaTb","author__title-link":"Author-module--author__title-link--Yrism",author__subtitle:"Author-module--author__subtitle--cAaEB"}},je8k:function(e,t,a){e.exports={menu:"Menu-module--menu--Efbin",menu__list:"Menu-module--menu__list--31Zeo","menu__list-item":"Menu-module--menu__list-item--1lJ6B","menu__list-item-link":"Menu-module--menu__list-item-link--10Ush","menu__list-item-link--active":"Menu-module--menu__list-item-link--active--2CbUO"}}}]);
//# sourceMappingURL=component---src-templates-page-template-js-5bb4cb493d6d64cb3007.js.map